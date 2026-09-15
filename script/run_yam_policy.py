#!/usr/bin/env python3
"""Run the LingBot-VA policy on real YAM arms.

Closes the loop between two servers:

    raiden `rd serve --action-type joint`   cameras + measured joints, and the arms
    wan_va policy server (CONFIG_NAME=yam)  predicts EEF-pose action chunks

and converts between them with `yam_bridge.action_convert`, which turns the model's
relative end-effector poses into absolute YAM joint commands via IK.

Run in the raiden venv, which has chiral, mujoco/mink and websockets::

    /data2/takuya/raiden/.venv/bin/python script/run_yam_policy.py --prompt "<task>"

Two properties of this system shape everything below:

  - **Inference cannot overlap execution.** The next `infer` needs a world model
    advanced by observations captured *during* the previous chunk, so the stages are
    strictly sequential. The arms hold position while the model thinks; inference
    latency sets the duty cycle, and `--dry-run` is the way to measure it safely.
  - **The policy never sees the robot.** It is fed back the chunk it just emitted, not
    measured state, so IK residual, joint clipping and tracking error are invisible to
    it. The gates and the tracking monitor here are the only things watching.
"""
import argparse
import os
import signal
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yam_bridge.action_convert import (  # noqa: E402
    BIMANUAL_DOF, DEFAULT_I2RT_ROOT, YamActionConverter,
)
from yam_bridge.va_client import VAClient, VAServerDied  # noqa: E402

# raiden camera name -> LingBot-VA observation key. Identical to
# `DEFAULT_CAMERA_MAP` in raiden/lerobot_export.py:59-63, which is the mapping the
# training dataset was built with. Order matters: index 0 is the full-resolution
# camera in the server's T-shape mosaic.
CAMERA_MAP = {
    "scene_camera": "observation.images.cam_high",
    "left_wrist_camera": "observation.images.cam_left_wrist",
    "right_wrist_camera": "observation.images.cam_right_wrist",
}
PROPRIO_KEYS = ("follower_l_joint_pos", "follower_r_joint_pos")


def build_obs(observation) -> Dict[str, np.ndarray]:
    """chiral `Observation` -> the dict `_encode_obs` expects (uint8 HWC RGB)."""
    out = {}
    for cam in observation.cameras:
        key = CAMERA_MAP.get(cam.name)
        if key is not None:
            out[key] = np.ascontiguousarray(cam.image)
    missing = set(CAMERA_MAP.values()) - set(out)
    if missing:
        raise RuntimeError(
            f"missing cameras {sorted(missing)}; raiden served "
            f"{[c.name for c in observation.cameras]}. The server keys off the "
            "camera.json key names, not the 'role' field.")
    return out


def measured_joints(observation) -> np.ndarray:
    """(14,) measured joints, left then right -- the layout raiden's consumers use."""
    try:
        parts = [np.asarray(observation.proprios[k], dtype=np.float64)
                 for k in PROPRIO_KEYS]
    except KeyError as exc:
        raise RuntimeError(
            f"proprio {exc} missing; is rd serve running with both followers?") from exc
    q14 = np.concatenate(parts)
    if q14.shape[0] != BIMANUAL_DOF:
        raise RuntimeError(f"expected 14 joint values, got {q14.shape[0]}")
    return q14


class Stopper:
    """Ctrl-C once to finish the current chunk cleanly; twice to give up."""

    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        if self.stop:
            print("\n[stop] second interrupt -- exiting immediately", flush=True)
            raise KeyboardInterrupt
        self.stop = True
        print("\n[stop] finishing the current chunk, then holding...", flush=True)


def run_episode(robot, va: VAClient, conv: YamActionConverter, prompt: str,
                args, stopper: Stopper) -> dict:
    """One rollout. Returns a summary dict."""
    # chiral multiplexes one socket with no request IDs, so a one-shot call racing the
    # background stream can be handed the wrong response. Keep the stream stopped
    # around reset(), which is a one-shot round trip (and homes the arms, ~2s).
    robot.stop_obs_stream()
    observation, _ = robot.reset()

    q14 = measured_joints(observation)
    conv.reset_stats()
    conv.latch_anchor(q14)
    print(f"[reset] anchored at home, q14[:6]={np.round(q14[:6], 3)}")

    va.reset(prompt)
    first_obs = build_obs(observation)

    robot.start_obs_stream(hz=args.obs_hz)
    robot.start_action_dispatch(hz=args.control_hz)

    period = 1.0 / args.control_hz
    chunk_i = 0
    first = True
    executed = 0
    halt_reason = None

    while chunk_i < args.max_chunks and not stopper.stop:
        t_infer = time.perf_counter()
        action = va.infer(first_obs)                    # (16, F, N)
        infer_ms = (time.perf_counter() - t_infer) * 1e3

        if action.ndim != 3 or action.shape[0] != 16:
            raise RuntimeError(f"unexpected action shape {action.shape}")
        n_per_frame = action.shape[2]
        if n_per_frame % 4 != 0:
            raise RuntimeError(f"action_per_frame {n_per_frame} is not a multiple of 4")
        steps_per_keyframe = n_per_frame // 4

        # Latent frame 0 of the first chunk is the conditioning frame rebuilt from the
        # initial observation; its actions are the zero/anchor action and executing
        # them would lurch the arms. `state` below still gets the full array.
        start_idx = 1 if first else 0

        t_ik = time.perf_counter()
        before_unsafe = conv.stats["unsafe_jumps"]
        cmds = conv.convert_chunk(action[:, start_idx:, :])
        ik_ms = (time.perf_counter() - t_ik) * 1e3

        if conv.stats["unsafe_jumps"] > before_unsafe:
            # An IK seed stalled and was rescued from a fallback seed, which switches
            # solution branch. Executing that step is a large slew that raiden's
            # _check_joint_delta would emergency_stop on.
            halt_reason = (f"IK branch switch of "
                           f"{conv.stats['max_fallback_jump_rad']:.3f} rad "
                           f"(> {conv.max_jump_rad}); holding instead of slewing")
            break

        keyframes: List[Dict[str, np.ndarray]] = []
        deadline = time.perf_counter()
        max_track = 0.0

        for i, cmd in enumerate(cmds):
            if not args.dry_run:
                robot.put_action(cmd.astype(np.float32))
            executed += 1
            deadline += period
            time.sleep(max(0.0, deadline - time.perf_counter()))

            if (i + 1) % steps_per_keyframe == 0:
                obs = robot.latest_obs
                if obs is None:
                    halt_reason = "no observation from raiden yet"
                    break
                keyframes.append(build_obs(obs))
                # Meaningless under --dry-run: nothing was commanded, so the arm sits
                # still while the commands advance and this would always trip.
                if not args.dry_run:
                    track = float(np.abs(measured_joints(obs)[:6] - cmd[:6]).max())
                    max_track = max(max_track, track)
                    if track > args.max_tracking_error:
                        halt_reason = (f"tracking error {track:.3f} rad exceeds "
                                       f"{args.max_tracking_error} -- the arm is not "
                                       "following; collision or torque limit?")
                        break

        if halt_reason is not None:
            break

        # A ragged keyframe list desyncs the server's streaming VAE, so only advance
        # the world model when the chunk completed intact.
        expected = (action.shape[1] - start_idx) * 4
        if len(keyframes) != expected:
            halt_reason = (f"got {len(keyframes)} keyframes, expected {expected}; "
                           "skipping compute_kv_cache to avoid desyncing the VAE")
            break
        va.advance(keyframes, action)

        duty = len(cmds) * period / max(1e-6, len(cmds) * period + infer_ms / 1e3)
        print(f"[chunk {chunk_i:3d}] infer {infer_ms:7.0f}ms  ik {ik_ms:5.1f}ms  "
              f"{len(cmds):2d} steps  {len(keyframes)} keyframes  "
              f"duty {duty * 100:4.1f}%  track {max_track:.3f} rad", flush=True)

        first = False
        chunk_i += 1

    return {
        "chunks": chunk_i,
        "steps": executed,
        "halt_reason": halt_reason,
        "stats": dict(conv.stats),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="Task instruction for the policy.")
    ap.add_argument("--robot-uri", default="ws://localhost:8765",
                    help="chiral PolicyClient URI for rd serve.")
    ap.add_argument("--va-host", default="0.0.0.0")
    ap.add_argument("--va-port", type=int, default=29536)
    ap.add_argument("--i2rt-root", default=DEFAULT_I2RT_ROOT)
    ap.add_argument("--control-hz", type=float, default=30.0,
                    help="Must match rd serve --control-hz. The model's actions are "
                         "sampled at the dataset's 30fps, so 30 executes them at the "
                         "rate they were trained at.")
    ap.add_argument("--obs-hz", type=float, default=10.0,
                    help="Observation stream rate. Keyframes are only needed every "
                         "200ms; higher just costs bandwidth (~2.8MB per frame at "
                         "480x640 x3 cameras, uncompressed).")
    ap.add_argument("--max-chunks", type=int, default=40,
                    help="Chunk budget per episode (~1.6s of motion each).")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-jump-rad", type=float, default=0.2,
                    help="IK branch-switch threshold; above this the chunk is refused.")
    ap.add_argument("--max-tracking-error", type=float, default=0.25,
                    help="Halt if measured joints diverge from commanded by more than "
                         "this. The policy cannot see this divergence itself.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the full loop but never send actions to the robot. Still "
                         "reads real observations, so it measures inference latency and "
                         "exercises the whole protocol with the arms untouched.")
    args = ap.parse_args()

    from chiral import PolicyClient  # noqa: E402  (raiden venv only)

    conv = YamActionConverter(args.i2rt_root, max_jump_rad=args.max_jump_rad)
    va = VAClient(host=args.va_host, port=args.va_port)
    stopper = Stopper()

    if args.dry_run:
        print("[dry-run] actions will NOT be sent to the robot")

    # One connection for the whole session: disconnecting triggers raiden's
    # emergency_stop(), which homes the arms and calls os._exit(0) on the server.
    with PolicyClient(args.robot_uri) as robot:
        meta = robot.get_metadata()
        shape = list(meta.get("action_shape", []))
        if meta.get("action_type") != "joint" or shape != [BIMANUAL_DOF]:
            raise SystemExit(
                f"rd serve is in {meta.get('action_type')!r} mode with action_shape "
                f"{shape}; this client sends 14-D joint commands. "
                "Relaunch with --action-type joint.")
        print(f"[robot] connected: cameras={meta.get('cameras')}")

        for ep in range(args.episodes):
            if stopper.stop:
                break
            print(f"\n=== episode {ep + 1}/{args.episodes}: {args.prompt!r} ===")
            try:
                summary = run_episode(robot, va, conv, args.prompt, args, stopper)
            except VAServerDied as exc:
                print(f"\n[fatal] policy server died and must be relaunched:\n{exc}")
                return 1
            finally:
                robot.stop_action_dispatch()

            print(f"[episode {ep + 1}] {summary['chunks']} chunks, "
                  f"{summary['steps']} steps executed")
            if summary["halt_reason"]:
                print(f"[episode {ep + 1}] HALTED: {summary['halt_reason']}")
            s = summary["stats"]
            print(f"  IK: {s['ik_failures']} failures, {s['ik_fallbacks']} fallbacks, "
                  f"max residual {s['max_pos_residual_m'] * 1000:.2f}mm / "
                  f"{s['max_ori_residual_rad']:.1e}rad, "
                  f"gripper clipped {s['gripper_clipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
