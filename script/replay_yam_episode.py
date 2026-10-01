#!/usr/bin/env python3
"""Replay a recorded `cover_blocks` episode on real YAM arms, with no policy involved.

This is verification step 2 of the deployment sequence: *is the hardware path safe?*
It isolates the 30 Hz transport, raiden's `_smooth_command` interpolation and the
`--max-joint-delta` threshold from policy quality, by sending a trajectory that is
known-good -- one the arms already executed during recording.

Two sources, in increasing order of what they exercise::

    --source raw        the recorded joint commands, sent verbatim.
                        Tests transport and pacing only. Run this first.
    --source converted  eef16 --relativize--> converter --IK--> joints14.
                        Adds the deployment converter, so it is the trajectory
                        shape the policy path will actually produce.

The number this run exists to produce is **commanded-vs-measured joint delta**, the
quantity `_check_joint_delta` tests (`raiden/server.py:993-1005`) and the one that
calls `emergency_stop()` on exceedance. Offline validation could not measure it: it
is tracking lag under real load, not the step-to-step delta of the command stream.
`script/validate_yam_action_convert.py` reported p99.99 = 0.234 rad for the latter;
this reports the former, which is what `--max-joint-delta` must actually clear.

Run in the raiden venv, which has chiral, mujoco/mink and websockets::

    # dataset-only, no robot, no rd serve -- run this first, it is free
    /data2/takuya/raiden/.venv/bin/python script/replay_yam_episode.py --no-robot

    # against hardware
    rd serve --action-type joint --control-hz 30 --max-joint-delta 0.5 \
             --resize-images 480x640 --no-depth
    /data2/takuya/raiden/.venv/bin/python script/replay_yam_episode.py \
        --robot-uri ws://<robot-host>:8765 --episode 0 --max-steps 300

WARNING: disconnecting from `rd serve` triggers its `emergency_stop()`, which homes
the arms and calls `os._exit(0)`. The server must be relaunched after every run.
"""
import argparse
import os
import signal
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yam_bridge.action_convert import (  # noqa: E402
    BIMANUAL_DOF, DEFAULT_I2RT_ROOT, LEFT_ARM, RIGHT_ARM, YamActionConverter,
    joints14_to_eef16, relativize_action16,
)
from yam_bridge.robot_io import (  # noqa: E402
    ARM_JOINTS, check_server_metadata, measured_joints, pct, report_bandwidth,
    tracking_error,
)


def read_column(path: str, column: str) -> np.ndarray:
    """Read one FixedSizeList column of a LeRobot parquet as (N, D) float64."""
    table = pq.read_table(path, columns=[column])
    return np.stack(table.column(column).to_numpy(zero_copy_only=False)).astype(np.float64)


def episode_path(root: str, index: int) -> str:
    return os.path.join(root, "data", "chunk-000", f"episode_{index:06d}.parquet")


def build_approach(start_q14: np.ndarray, first_cmd: np.ndarray, steps: int) -> np.ndarray:
    """Linear ramp from the measured pose to the trajectory's first command.

    With `--anchor episode` the first command is the episode's own frame 0, which is
    near but not equal to the home pose raiden resets to, so something has to close
    that gap gradually. `--anchor robot` makes the ramp a no-op by construction (the
    first command *is* the measured pose), which costs nothing.
    """
    if steps <= 0:
        return np.zeros((0, BIMANUAL_DOF), dtype=np.float64)
    alphas = np.linspace(0.0, 1.0, steps + 1)[1:-1].reshape(-1, 1)
    return (1.0 - alphas) * start_q14.reshape(1, -1) + alphas * first_cmd.reshape(1, -1)


def build_commands(args, anchor_q14: np.ndarray) -> tuple:
    """Return `(cmds (T, 14), joints14, state14, converter or None)`.

    `anchor_q14` is the configuration the relative trajectory is composed onto, and
    which one you pick matters far more than it looks. `add_eef_pose` adds translation
    in the arm base frame, so changing the anchor translates the *entire* trajectory
    rigidly. Measured on episode 0: the arms home to within 0.013 rad of the episode's
    own frame 0, yet through FK that is a 14.4 mm shift on the right arm -- enough to
    push the reach-limit portion of the demonstration (target radius up to 0.56 m)
    ~1.8 mm outside the workspace. IK then misses its 1e-4 m tolerance on 449 right-arm
    steps and burns every fallback seed trying, at 52 ms/step instead of 1.3.

    Hence `--anchor episode` by default: a replay exists to reproduce a recorded
    trajectory, and the demonstrated anchor is the one that trajectory is valid for.
    `--anchor robot` reproduces what deployment does instead, and is the way to see
    this failure mode on purpose.
    """
    src = episode_path(args.src, args.episode)
    joints14 = read_column(src, "action")
    state14 = read_column(src, "observation.state")

    # Truncate before converting, not after: IK dominates the setup cost (a full
    # 1776-step episode is ~2 s healthy and ~100 s when it saturates), and stats
    # accumulated over steps that are never dispatched misreport the run.
    if args.max_steps:
        joints14 = joints14[:args.max_steps]
        state14 = state14[:args.max_steps]

    if args.source == "raw":
        return joints14.copy(), joints14, state14, None

    eef16 = read_column(episode_path(args.dst, args.episode), "action")
    if len(eef16) < len(joints14):
        raise SystemExit(f"length mismatch: {len(joints14)} joints vs {len(eef16)} eef")
    eef16 = eef16[:len(joints14)]

    conv = YamActionConverter(args.i2rt_root, max_jump_rad=args.max_jump_rad)
    conv.latch_anchor(anchor_q14)
    t0 = time.perf_counter()
    cmds = conv.convert_sequence(relativize_action16(eef16))
    dt = time.perf_counter() - t0
    print(f"[convert] {len(cmds)} steps in {dt:.1f}s ({dt / len(cmds) * 1e3:.2f} ms/step)")
    return cmds, joints14, state14, conv


def preflight(cmds: np.ndarray, joints14: np.ndarray, state14: np.ndarray,
              conv, start_q14: np.ndarray, args) -> list:
    """Gate the whole trajectory before a single action is sent. Returns failures.

    Everything here is checkable offline, and each check corresponds to a specific
    way the run would end badly on hardware rather than merely badly.
    """
    fails = []
    step = np.abs(np.diff(cmds[:, ARM_JOINTS], axis=0))
    first = np.abs(cmds[0, ARM_JOINTS] - start_q14[ARM_JOINTS]).max()
    # The demonstration's own commanded-vs-measured error. This is the tracking lag
    # the hardware showed while recording at its native rate, so it is the empirical
    # floor for --max-joint-delta: the threshold cannot be tighter than this and
    # still pass on a trajectory the arms are known to execute.
    recorded_track = np.abs(joints14[:, ARM_JOINTS] - state14[:, ARM_JOINTS])

    print(f"\n[preflight] {len(cmds)} steps = {len(cmds) / args.control_hz:.1f}s "
          f"at {args.control_hz:g} Hz, source={args.source}")
    print(f"  step delta (rad)    p50={pct(step, 50):.4f}  p99={pct(step, 99):.4f}  "
          f"p99.99={pct(step, 99.99):.4f}  max={step.max():.4f}")
    print(f"  recorded track err  p50={pct(recorded_track, 50):.4f}  "
          f"p99={pct(recorded_track, 99):.4f}  max={recorded_track.max():.4f}")
    print(f"  first cmd vs now    {first:.4f} rad")

    if conv is not None:
        s = conv.stats
        # The anchor's EEF offset from the demonstration's own starting pose is what
        # predicts IK trouble, and it is invisible in joint space: 0.013 rad of
        # difference at the shoulder is 14 mm at the gripper, and the whole trajectory
        # translates by it. Reported per arm because the two anchors differ
        # independently and the failures land on whichever arm moved further.
        ep_anchor = joints14_to_eef16(joints14[0], conv.kin_left, conv.kin_right)
        off_l = np.linalg.norm(conv.anchor16[LEFT_ARM][:3] - ep_anchor[LEFT_ARM][:3])
        off_r = np.linalg.norm(conv.anchor16[RIGHT_ARM][:3] - ep_anchor[RIGHT_ARM][:3])
        print(f"  anchor vs episode   L {off_l * 1000:.1f} mm  R {off_r * 1000:.1f} mm "
              f"(--anchor {args.anchor})")
        print(f"  IK                  {s['ik_failures']} failures, "
              f"{s['ik_fallbacks']} fallbacks, {s['unsafe_jumps']} unsafe jumps, "
              f"max residual {s['max_pos_residual_m'] * 1000:.2f} mm")
        if s["ik_failures"]:
            if max(off_l, off_r) > 1e-3 and s["max_pos_residual_m"] < 5e-3:
                # Small residual + an anchor offset is workspace saturation, not a
                # solver fault: IK reached the nearest point it could and missed the
                # 1e-4 m tolerance. Say so, because "IK failures" alone reads as a bug.
                hint = (" -- residual is only "
                        f"{s['max_pos_residual_m'] * 1000:.2f} mm, so these are "
                        "reach-limit misses caused by the "
                        f"{max(off_l, off_r) * 1000:.1f} mm anchor offset, not a "
                        "solver fault. Try --anchor episode.")
            else:
                hint = " -- commands are not on the demonstrated trajectory"
            fails.append(f"{s['ik_failures']} IK failures{hint}")
        # A fallback re-seed switches IK branch, a step change in joint space that
        # _check_joint_delta would emergency_stop the server on.
        if s["unsafe_jumps"]:
            fails.append(f"{s['unsafe_jumps']} IK branch switches "
                         f"(max {s['max_fallback_jump_rad']:.3f} rad)")

    # Not a hard gate against _check_joint_delta -- that compares against *measured*,
    # not the previous command -- but a step larger than the threshold cannot be
    # tracked within one control period, so it is guaranteed to trip it.
    if step.max() > args.max_joint_delta:
        fails.append(f"max step delta {step.max():.3f} rad exceeds --max-joint-delta "
                     f"{args.max_joint_delta} -- rd serve would e-stop")
    if first > args.max_joint_delta:
        fails.append(f"first command is {first:.3f} rad from the current pose; "
                     "the arm would lurch on the first step")
    return fails


def run(robot, cmds: np.ndarray, args) -> dict:
    """Dispatch the command stream at `control_hz`, watching tracking error."""
    period = 1.0 / args.control_hz
    robot.start_obs_stream(hz=args.obs_hz)
    # Tick the dispatcher faster than we enqueue, so *our* deadline-corrected loop
    # sets the cadence and the dispatcher only drains.
    #
    # chiral's `_aaction_dispatch` (client.py:268-280) sleeps `interval - send_time`
    # each pass, which compensates for the send but not for asyncio.sleep's own
    # overshoot -- it has no absolute deadline, so the overshoot accumulates. Measured
    # here: 0.72 ms per tick, i.e. 29.37 Hz when asked for 30, so the queue grows ~3
    # actions every 150 steps without bound. That backlog *is* latency: 31 queued
    # actions is a full second of commands the arm has not started yet, which is why
    # the best-fit observation lag walked out to 24 steps and the tracking error only
    # "failed" once the true lag exceeded the search window.
    #
    # Lowering the rate does not help: the error is per tick and an episode is the
    # same number of ticks at any rate, so 10 Hz accumulates the same ~1.3 s.
    robot.start_action_dispatch(hz=args.control_hz * args.dispatch_mult)

    track, raw_track, lags, jitter, backlog = [], [], [], [], []
    halt_reason = None
    executed = 0
    t_start = time.perf_counter()
    deadline = t_start

    for i, cmd in enumerate(cmds):
        if STOP["flag"]:
            halt_reason = "interrupted by user"
            break
        robot.put_action(cmd.astype(np.float32))
        executed += 1

        if i % args.check_every == 0:
            obs = robot.latest_obs
            if obs is None:
                if time.perf_counter() - t_start > args.obs_timeout:
                    halt_reason = "no observation from raiden"
                    break
            else:
                err, lag, raw = tracking_error(
                    cmds, i, measured_joints(obs), args.max_obs_lag)
                track.append(err)
                raw_track.append(raw)
                if lag is not None:
                    lags.append(lag)
                if err > args.max_tracking_error:
                    halt_reason = (f"tracking error {err:.3f} rad at step {i} exceeds "
                                   f"{args.max_tracking_error} even at best lag "
                                   f"{lag} -- the arm is not following; collision or "
                                   "torque limit?")
                    break
            queued = len(getattr(robot, "_action_queue", ()))
            backlog.append(queued)
            # Gate on backlog directly. It is a cleaner failure signal than tracking
            # error, because queued actions are pure latency -- the arm has not begun
            # them -- so a growing queue silently invalidates every other measurement
            # here before it ever looks like a tracking problem.
            if queued > args.max_backlog:
                halt_reason = (
                    f"action backlog {queued} > {args.max_backlog} at step {i}: the "
                    f"dispatcher is not keeping up, so commands are running "
                    f"{queued / args.control_hz:.2f}s behind and every latency figure "
                    "below is meaningless. Raise --dispatch-mult.")
                break

        deadline += period
        # Measure lateness *after* sleeping. Measuring the slack before the sleep only
        # reports how long the loop body took, which is why the first run showed a
        # constant -33 ms: that was one control period of slack, not early dispatch.
        time.sleep(max(0.0, deadline - time.perf_counter()))
        jitter.append(time.perf_counter() - deadline)

        if (i + 1) % (int(args.control_hz) * 5) == 0:
            print(f"  [{i + 1:5d}/{len(cmds)}] "
                  f"t={time.perf_counter() - t_start:6.1f}s  "
                  f"track={max(track[-20:], default=0.0):.3f} rad  "
                  f"backlog={backlog[-1] if backlog else 0}", flush=True)

    # Read the queue before stopping the dispatcher: what is still in it never
    # reached the robot, so the true send rate is executed - leftover.
    leftover = len(getattr(robot, "_action_queue", ()))
    robot.stop_action_dispatch()
    return {
        "executed": executed,
        "leftover": leftover,
        "elapsed": time.perf_counter() - t_start,
        "halt_reason": halt_reason,
        "track": np.asarray(track) if track else np.zeros(0),
        "raw_track": np.asarray(raw_track) if raw_track else np.zeros(0),
        "lags": np.asarray(lags) if lags else np.zeros(0),
        "jitter": np.asarray(jitter) if jitter else np.zeros(0),
        "backlog": np.asarray(backlog) if backlog else np.zeros(0),
    }


STOP = {"flag": False}


def _on_sigint(signum, frame):
    if STOP["flag"]:
        raise KeyboardInterrupt
    STOP["flag"] = True
    print("\n[stop] halting after the current step; arms hold position", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--src", default="Datasets/yam-real-world-lerobot/cover_blocks",
                    help="Raw joint-space LeRobot dataset (14-dim ground truth).")
    ap.add_argument("--dst", default="Datasets/yam-real-world-lingbot-va/cover_blocks",
                    help="Converted EEF dataset (16-dim).")
    ap.add_argument("--source", choices=("converted", "raw"), default="converted")
    ap.add_argument("--anchor", choices=("episode", "robot"), default="episode",
                    help="Pose the relative trajectory is composed onto. 'episode' "
                         "uses the demonstration's own frame 0 and reproduces the "
                         "recorded motion; 'robot' uses the measured pose at reset, "
                         "which is what deployment does -- see build_commands() for "
                         "why a 14 mm difference there costs 449 IK failures.")
    ap.add_argument("--i2rt-root", default=DEFAULT_I2RT_ROOT)
    ap.add_argument("--robot-uri",
                    help="chiral PolicyClient URI for rd serve, e.g. "
                         "ws://mlcav.seas.ucla.edu:8765. Required unless --no-robot. "
                         "Deliberately has no default: silently dialling localhost "
                         "when the arms are on another host wastes a launch cycle.")
    ap.add_argument("--control-hz", type=float, default=30.0,
                    help="Must match rd serve --control-hz. The recording is 30 fps, "
                         "so 30 replays it at the rate it was captured.")
    ap.add_argument("--obs-hz", type=float, default=30.0,
                    help="Observation poll rate. Polling at the control rate makes "
                         "every measurement at most one step old, which is what makes "
                         "the reported lag meaningful rather than an artefact of the "
                         "poll interval. Only affordable with small images -- see the "
                         "bandwidth estimate printed after reset. The replay reads "
                         "nothing but proprios, so serve it "
                         "`rd serve --resize-images 64x64` and this is free.")
    ap.add_argument("--check-every", type=int, default=3,
                    help="Sample tracking error every N steps (3 = 10 Hz at 30 Hz "
                         "control). Cheap, and more samples sharpen the lag estimate.")
    ap.add_argument("--dispatch-mult", type=float, default=4.0,
                    help="Run chiral's action dispatcher at this multiple of "
                         "--control-hz. Its loop sleeps a fixed interval rather than "
                         "tracking an absolute deadline, so asking it for exactly "
                         "--control-hz runs ~2%% slow and the queue grows without "
                         "bound. Over-ticking makes it drain on demand and lets this "
                         "script's deadline-corrected loop set the real cadence; the "
                         "cost is up to 1/(control_hz*mult) of send jitter.")
    ap.add_argument("--max-backlog", type=int, default=10,
                    help="Halt if this many actions are queued but unsent. Backlog is "
                         "latency the arm has not begun working through, and it "
                         "invalidates every timing measurement in the report.")
    ap.add_argument("--max-obs-lag", type=int, default=24,
                    help="How many recent commands to search when matching a "
                         "measurement (24 = 800ms at 30 Hz). Must comfortably exceed "
                         "the real pipeline latency or tracking error reads high; the "
                         "report says so if the best lag hits this limit. Offline "
                         "replay of the first hardware run put the true lag at 13-15 "
                         "steps, which is poll interval + transport + actuator lag.")
    ap.add_argument("--obs-timeout", type=float, default=5.0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="Truncate the episode (0 = whole thing, ~40-70s).")
    ap.add_argument("--approach-steps", type=int, default=45,
                    help="Steps to ramp from the measured pose to the trajectory's "
                         "first command (45 = 1.5s at 30 Hz). Needed for "
                         "--source raw and --anchor episode; a no-op for "
                         "--anchor robot, whose first command is the measured pose.")
    ap.add_argument("--max-joint-delta", type=float, default=0.5,
                    help="Mirror of rd serve's flag, checked client-side in preflight "
                         "so a trajectory that would e-stop never gets dispatched. "
                         "0.5 because _check_joint_delta compares against *measured* "
                         "joints, and the recorded demonstrations themselves reach "
                         "0.39 rad of tracking lag (p99.99 = 0.29).")
    ap.add_argument("--max-jump-rad", type=float, default=0.2,
                    help="IK branch-switch threshold for the converter.")
    ap.add_argument("--max-tracking-error", type=float, default=0.45,
                    help="Halt if measured diverges from commanded by more than this. "
                         "Just under --max-joint-delta, so the client stops before "
                         "rd serve e-stops itself.")
    ap.add_argument("--no-robot", action="store_true",
                    help="Dataset-only: build the commands, run every preflight gate "
                         "and print the budget report, without connecting to raiden.")
    ap.add_argument("--force", action="store_true",
                    help="Dispatch even if preflight fails. Only with a reason.")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)

    episode_q0 = read_column(episode_path(args.src, args.episode), "action")[0]

    if args.no_robot:
        if args.anchor == "robot":
            raise SystemExit("--anchor robot needs a live reset to read the measured "
                             "pose from; use --anchor episode with --no-robot.")
        cmds, joints14, state14, conv = build_commands(args, episode_q0)
        # Stand in for the measured pose with the episode's own start, so the approach
        # ramp and the first-command gate are exercised the way they will be on
        # hardware. The arms home to within ~0.013 rad of this.
        cmds = np.concatenate([build_approach(episode_q0, cmds[0], args.approach_steps),
                               cmds])
        fails = preflight(cmds, joints14, state14, conv, episode_q0, args)
        print()
        for f in fails:
            print(f"  FAIL: {f}")
        print("[no-robot] preflight " + ("FAILED" if fails else "passed")
              + "; nothing was sent.")
        return 1 if fails else 0

    if not args.robot_uri:
        raise SystemExit("--robot-uri is required (or pass --no-robot). Example: "
                         "--robot-uri ws://mlcav.seas.ucla.edu:8765")

    from chiral import PolicyClient  # noqa: E402  (raiden venv only)

    with PolicyClient(args.robot_uri) as robot:
        meta = robot.get_metadata()
        print(f"[robot] connected: {meta.get('cameras')}, "
              f"control_hz={meta.get('control_hz')}, "
              f"max_joint_delta={meta.get('max_joint_delta')}")
        fails = check_server_metadata(meta, args.control_hz, args.max_joint_delta,
                                      BIMANUAL_DOF)
        if fails:
            raise SystemExit("\n".join("ERROR: " + f for f in fails))

        # chiral multiplexes one socket with no request IDs, so a one-shot call racing
        # the background stream can be handed the wrong response. reset() is one-shot
        # (and homes the arms, ~2s), so keep the stream stopped around it.
        robot.stop_obs_stream()
        observation, _ = robot.reset()
        start_q14 = measured_joints(observation)
        # All 14, not q14[:6]: the right arm and both grippers are needed to reproduce
        # a run offline, and the right arm is where anchor-offset trouble showed up.
        print(f"[reset] homed, measured q14 = {np.round(start_q14, 4)}")
        report_bandwidth(observation, args.obs_hz, need_images=False)

        anchor = episode_q0 if args.anchor == "episode" else start_q14
        cmds, joints14, state14, conv = build_commands(args, anchor)
        cmds = np.concatenate([build_approach(start_q14, cmds[0], args.approach_steps),
                               cmds])
        fails = preflight(cmds, joints14, state14, conv, start_q14, args)
        if fails:
            for f in fails:
                print(f"  FAIL: {f}")
            if not args.force:
                print("[abort] preflight failed; nothing sent. Re-run with --force to "
                      "override, or with --source raw to isolate the converter.")
                return 1
            print("[force] dispatching despite preflight failure")

        input_ok = args.force or input("\nMove the arms? Clear the workspace first. "
                                       "[y/N] ").strip().lower() == "y"
        if not input_ok:
            print("[abort] not dispatching.")
            return 1

        # The demonstration's own command-vs-measurement error, on this hardware.
        # It is the closest thing to what rd serve's _check_joint_delta actually
        # sees, so the threshold suggestion below is sized against it.
        recorded_track_max = float(
            np.abs(joints14[:, ARM_JOINTS] - state14[:, ARM_JOINTS]).max())
        r = run(robot, cmds, args)

    print(f"\n[done] {r['executed']}/{len(cmds)} steps in {r['elapsed']:.1f}s "
          f"(expected {len(cmds) / args.control_hz:.1f}s)")
    if r["halt_reason"]:
        print(f"[done] HALTED: {r['halt_reason']}")
    if r["track"].size:
        t, raw, lg = r["track"], r["raw_track"], r["lags"]
        print(f"  tracking err (rad)  p50={pct(t, 50):.4f}  p99={pct(t, 99):.4f}  "
              f"max={t.max():.4f}   <- lag-compensated: true tracking accuracy")
        print(f"  uncompensated       p50={pct(raw, 50):.4f}  p99={pct(raw, 99):.4f}  "
              f"max={raw.max():.4f}   (inflated by observation latency -- ignore)")
        if lg.size:
            print(f"  best-fit obs lag    p50={pct(lg, 50):.0f}  p90={pct(lg, 90):.0f} "
                  f"steps = {pct(lg, 90) / args.control_hz * 1e3:.0f} ms, from "
                  f"{lg.size}/{t.size} samples where the arm was moving enough to "
                  "localise it")
            if pct(lg, 90) >= args.max_obs_lag:
                print("  NOTE: best lag hit the search limit -- raise --max-obs-lag; "
                      "the compensated figure above is still contaminated.")
        else:
            print(f"  best-fit obs lag    not identifiable in any of {t.size} samples "
                  "-- the arm barely moved")

        # The lag-compensated figure is NOT the right basis for --max-joint-delta.
        # _check_joint_delta compares a just-issued command against a fresh (~10 ms)
        # measurement, so it sees the arm's actuator lag, which is exactly what
        # compensation removes. The recorded demonstration is the honest proxy: its
        # `action` vs `observation.state` is a command-vs-fresh-measurement pair on
        # this same hardware. Size the threshold off whichever is larger.
        basis = max(t.max(), recorded_track_max)
        print(f"  suggested --max-joint-delta {max(0.1, basis * 1.5):.2f}  "
              f"(1.5x the larger of measured {t.max():.3f} and the demonstration's "
              f"own {recorded_track_max:.3f}; the latter usually wins because the "
              "server's check is not lag-compensated)")
    if r["jitter"].size:
        j = r["jitter"] * 1e3
        print(f"  dispatch lateness   p50={pct(j, 50):+.2f}ms  p99={pct(j, 99):+.2f}ms  "
              f"max={j.max():+.2f}ms  (after sleep; >0 means behind schedule)")
    if r["backlog"].size:
        sent = r["executed"] - r["leftover"]
        print(f"  action backlog      max={int(r['backlog'].max())}  "
              f"final={r['leftover']}  (queued = not yet sent = pure added latency)")
        print(f"  effective send rate {sent / max(r['elapsed'], 1e-9):.2f} Hz vs "
              f"{args.control_hz:g} requested  "
              f"({args.control_hz * args.dispatch_mult:g} Hz dispatcher ticks)")
    print("\nNote: the connection just closed, so rd serve has emergency_stopped and "
          "exited. Relaunch it before the next run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
