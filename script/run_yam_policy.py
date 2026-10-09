#!/usr/bin/env python3
"""Run the LingBot-VA policy on real YAM arms.

Closes the loop between two servers:

    raiden `rd serve --action-type joint`   cameras + measured joints, and the arms
    wan_va policy server (CONFIG_NAME=yam_place_cube)  predicts EEF-pose action chunks

and converts between them with `yam_bridge.action_convert`, which turns the model's
relative end-effector poses into absolute YAM joint commands via IK. A checkpoint
trained with `action_space = 'joint'` (e.g. CONFIG_NAME=yam_place_cube_joint) emits
absolute joints instead; the server says so in its handshake, and the client then
only clips them to the joint limits -- no IK, so no IK branch-switch gate. Everything
else here (slew limit, tracking, artifacts) is identical, and the plots stay in EEF
space via FK so the two kinds of checkpoint compare directly.

Run in the raiden venv, which has chiral, mujoco/mink and websockets. All three
processes, in order::

    # 1. policy server (listens on 29536 -- run_launch_va_server_sync.sh defines PORT
    #    but never forwards it). Point --save_root at scratch: _reset writes
    #    latents_*.pt / actions_*.pt / obs_data_*.pt per step, unbounded.
    NGPU=1 CONFIG_NAME=yam_place_cube bash script/run_launch_va_server_sync.sh

    # 2. robot server. 480x640 is the resolution the policy was trained at, and
    #    --max-joint-delta 0.35 is measured, not guessed: the recorded demonstrations
    #    themselves reach 0.234 rad of commanded-vs-measured lag, and
    #    _check_joint_delta compares against measured joints.
    rd serve --action-type joint --control-hz 30 --max-joint-delta 0.35 \
             --resize-images 480x640 --no-depth

    # 3. this client -- start with --dry-run, which drives the whole protocol and
    #    measures inference latency without moving anything.
    /data2/takuya/raiden/.venv/bin/python script/run_yam_policy.py \
        --prompt "<task>" --robot-uri ws://<robot-host>:8765 --dry-run

`--save-artifacts` writes a trajectory plot and results.json per rollout under `--out`,
which defaults to the served checkpoint's position mirrored under eval_out/, e.g.
`eval_out/yam_place_cube/<run>/checkpoints/checkpoint_step_N/closed_loop/`. The leaf
directory is what separates this from an open-loop evaluation of the same checkpoint.
`--video` adds the imagined-vs-live comparison mp4. The plot carries three series --
what the model predicted, what IK and the slew limiter actually commanded, and what
the arms measurably did -- so a divergence can be attributed rather than merely
noticed. Both are written even when a safeguard halts the rollout, which is when
they are worth the most::

    ... --dry-run --save-artifacts --video

ONE rollout per invocation, because nothing here can tell when the task is done.
The policy never sees the robot and there is no success detector, so `--max-chunks`
is only a ceiling -- the operator watching the arms is the judge. Press Ctrl-C when
the task is finished (or clearly is not): the current chunk completes, the arms stop,
the artifacts are written and the process exits 0. That is the normal ending, not an
error path. A second Ctrl-C exits immediately and still writes the artifacts.

Because each run is one rollout, they are numbered per checkpoint and CONTINUE across
invocations: run this three times against one checkpoint and get ep0, ep1, ep2, with
results.json extended rather than replaced. A real-robot rollout cannot be reproduced,
so nothing here is ever allowed to overwrite one.

Startup homes the arms (`robot.reset()`) and then blocks until you press Enter, which
is when to place the objects. The prompt sits after homing and before the action
dispatcher starts -- the only point where the arms are at a known pose and cannot
move, so it is safe to reach in. Skipped when stdin is not a terminal. The first
observation and the rollout anchor are captured after Enter, so the policy starts from
the staged scene rather than from the one reset() saw.

Measured samples are recorded only where `tracking_error` can identify a lag. Across a
dwell every command in the window matches equally well, so the sample says nothing about
where in the trajectory the arm is; plotting it would draw a plateau that looks like
tracking data and is not. They are NOT lag-compensated, so a healthy arm reads ~5-6 steps
behind the command.

The hardware path underneath this is certified separately by
`script/replay_yam_episode.py`, which replays a recorded episode through the same
converter and transport with no policy involved. Run that first on any new machine:
if it cannot hold 30 Hz, neither can this.

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
import contextlib
import json
import os
import signal
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yam_bridge.action_convert import (  # noqa: E402
    BIMANUAL_DOF, DEFAULT_I2RT_ROOT, YamActionConverter, joints14_to_eef16,
    make_converter, relativize_action16,
)
from yam_bridge.eval_meta import (  # noqa: E402
    artifact_stem, eval_out_dir, next_rollout_index, resolve_served,
)
from yam_bridge.eval_viz import (  # noqa: E402
    ARMS, plot_trajectories, rotation_error_deg, save_imagination,
)
from yam_bridge.robot_io import (  # noqa: E402
    build_obs, check_server_metadata, measured_joints, pct, report_bandwidth,
    slew_limit, tracking_error,
)
from yam_bridge.va_client import VAClient, VAServerDied  # noqa: E402


class Stopper:
    """Ctrl-C once to finish the current chunk cleanly; twice to give up.

    One press is how a rollout normally ends -- the operator deciding the task is over
    -- so it must leave a complete record, not a truncated one. Hence the first press
    only sets a flag and the control loop exits at the next chunk boundary, rather than
    raising from wherever the arms happened to be mid-dispatch.
    """

    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        if self.stop:
            print("\n[stop] second interrupt -- exiting immediately", flush=True)
            raise KeyboardInterrupt
        self.stop = True
        print("\n[stop] finishing the current chunk, then holding...", flush=True)


@contextlib.contextmanager
def sigint_deferred(what: str):
    """Ignore Ctrl-C for the duration of a short, must-not-be-half-done step.

    Writing the artifacts is the last thing a rollout does, and it is reached from a
    `finally` -- so it runs *after* the user has already pressed Ctrl-C, often twice.
    Without this the second press lands inside matplotlib and destroys the very record
    the first press was trying to end the run with. The window is one figure and one
    json, well under a second, and the arms are already stopped before it opens.
    """
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        print(f"  [{what}] done; Ctrl-C active again", flush=True)


def to_anchor_relative(abs16: np.ndarray, anchor16: np.ndarray) -> np.ndarray:
    """(T,16) absolute EEF -> relative to the rollout anchor, the model's frame.

    The model emits poses relative to the anchor latched at reset, while FK of
    commanded or measured joints is absolute. Prepending the anchor and reusing
    `relativize_action16` -- which relativizes against a sequence's own first row --
    expresses both in one frame without restating the convention here, where it could
    drift from `add_eef_pose`'s inverse and silently offset the whole plot.
    """
    if not len(abs16):
        return np.zeros((0, 16))
    stacked = np.concatenate([anchor16[None, :], np.asarray(abs16)], axis=0)
    return relativize_action16(stacked)[1:]


def new_record() -> dict:
    """Empty accumulator for one rollout, owned by the CALLER.

    Owned by the caller on purpose. A second Ctrl-C raises out of `run_episode` from
    wherever it was blocked, and anything held only in that frame's locals dies with
    it -- which is how an interrupted rollout used to produce no artifact at all. The
    lists here are mutated in place, so whatever had been collected by the moment of
    the interrupt is still in the caller's hands.
    """
    return {
        "rollout": 0,      # index within the checkpoint's whole history, not this run
        "chunks": 0, "steps": 0, "halt_reason": None, "stats": {}, "lags": [],
        "track": [],       # lag-compensated tracking error per keyframe, rad
        "keyframe_gaps_ms": [],  # capture-time spacing of consecutive keyframes
        "keyframe_nominal_ms": None,  # the spacing training used
        "bounds": [],
        "pred_rel": [],    # model output, already anchor-relative
        "cmd_abs": [],     # post-IK, post-slew joints -> FK -> absolute
        "meas_abs": [],    # measured joints -> FK -> absolute
        "meas_steps": [],  # control step each measured sample belongs to
        "videos": [],
    }


def finalize_record(record: dict, conv: YamActionConverter) -> dict:
    """Turn the raw accumulators into the plottable arrays.

    Separate from `run_episode` so it can run from a `finally` after an interrupt,
    against however much was collected before it.
    """
    record["stats"] = dict(conv.stats)
    pred = (np.concatenate(record["pred_rel"]) if record["pred_rel"]
            else np.zeros((0, 16)))
    anchor = conv.anchor16
    if anchor is None:          # interrupted before the anchor was latched
        record.update({"predicted": np.zeros((0, 16)), "commanded": np.zeros((0, 16)),
                       "measured": np.zeros((0, 16)), "measured_steps": []})
        return record
    cmd = to_anchor_relative(np.asarray(record["cmd_abs"]), anchor)
    # Truncated to what was dispatched. Both series are appended a whole chunk at a
    # time, before its first step goes out, so a safeguard halting mid-chunk -- or a
    # second Ctrl-C -- leaves them holding steps that never ran, and plotting or
    # scoring that tail would report motion the arms never made.
    n = min(len(pred), len(cmd), record["steps"])
    meas = to_anchor_relative(np.asarray(record["meas_abs"]), anchor)
    keep = [i for i, s in enumerate(record["meas_steps"]) if s < n]
    record.update({
        "predicted": pred[:n],
        "commanded": cmd[:n],
        "measured": meas[keep] if len(meas) else np.zeros((0, 16)),
        "measured_steps": [record["meas_steps"][i] for i in keep],
    })
    return record


def wait_for_scene_reset(rollout: int, stopper: Stopper) -> bool:
    """Block until the operator has staged the workspace. False means give up.

    Called after `robot.reset()` has homed the arms and before the action dispatcher is
    started -- the one window in a run where the arms are both at a known pose and
    guaranteed not to move, so it is the safe moment to reach in and place objects.

    Skipped when stdin is not a terminal: a backgrounded or piped run would otherwise
    block forever on a prompt nobody can see.
    """
    if not sys.stdin.isatty():
        print("[pause] stdin is not a tty -- not waiting for a scene reset")
        return True
    try:
        input(f"[pause] arms are home. Stage the scene for rollout {rollout}, "
              f"then press Enter to start (Ctrl-C then Enter to abort): ")
    except EOFError:
        print("\n[pause] stdin closed -- continuing")
    # The first Ctrl-C only sets the flag (Stopper never raises on it), so a prompt
    # interrupted that way still needs its Enter before this returns.
    return not stopper.stop


def run_episode(robot, va: VAClient, conv: YamActionConverter, prompt: str,
                args, stopper: Stopper, record: dict) -> dict:
    """One rollout, accumulating into the caller-owned `record`."""
    # Drop anything the previous episode left queued. `stop_action_dispatch` only
    # cancels the dispatch task -- nothing in chiral ever clears `_action_queue`
    # (client.py:68, 195, 265) -- so commands still queued when an episode halts
    # survive into the next one. reset() homes the arms, then start_action_dispatch
    # below would immediately drain those stale commands and slew the arms from home
    # back to wherever the last episode stopped. `_check_joint_delta` would probably
    # e-stop on it, which is the safety net doing its job rather than a design.
    queue = getattr(robot, "_action_queue", None)
    if queue is not None and len(queue):
        print(f"[reset] discarding {len(queue)} stale queued actions")
        queue.clear()

    # chiral multiplexes one socket with no request IDs, so a one-shot call racing the
    # background stream can be handed the wrong response. Keep the stream stopped
    # around reset(), which is a one-shot round trip (and homes the arms, ~2s).
    robot.stop_obs_stream()
    observation, _ = robot.reset()

    home_q14 = measured_joints(observation)
    conv.reset_stats()
    print(f"[reset] homed, measured q14 = {np.round(home_q14, 4)}")
    report_bandwidth(observation, args.obs_hz, need_images=True)

    # Arms homed, dispatcher not yet started: hold here for the operator.
    if not wait_for_scene_reset(record["rollout"], stopper):
        record["halt_reason"] = "aborted before the rollout started"
        return record

    # Re-observe AFTER the operator has staged the scene. reset()'s observation shows
    # the workspace as it was before objects were placed, and chunk 0 conditions on
    # first_obs -- so reusing it would plan the first chunk, and seed the world model,
    # against a scene that no longer exists. One-shot is safe here for the same reason
    # it is around reset(): the stream is still stopped. The anchor comes from this
    # same observation, so image and pose describe one instant -- the pairing training
    # sees at episode frame 0 -- and it is latched next to va.reset(), which is where
    # latch_anchor() says it belongs.
    observation = robot.get_obs()
    first_obs = build_obs(observation)
    q14 = measured_joints(observation)
    conv.latch_anchor(q14)
    print(f"[start] anchored on the staged scene, measured q14 = {np.round(q14, 4)} "
          f"(max drift since homing {np.abs(q14 - home_q14).max():.4f} rad)")
    va.reset(prompt)

    robot.start_obs_stream(hz=args.obs_hz)
    # Tick the dispatcher faster than we enqueue so *our* deadline-corrected loop sets
    # the cadence and the dispatcher only drains. chiral's `_aaction_dispatch`
    # (client.py:268-280) sleeps `interval - send_time` with no absolute deadline, so
    # asyncio.sleep's overshoot accumulates: measured at 0.72 ms/tick, i.e. 29.37 Hz
    # when asked for 30, growing the queue ~3 actions every 150 steps without bound.
    # That backlog is pure latency, and lowering the rate does not help -- the error is
    # per tick and an episode is the same number of ticks at any rate.
    robot.start_action_dispatch(hz=args.control_hz * args.dispatch_mult)

    period = 1.0 / args.control_hz
    chunk_i = 0
    first = True
    executed = 0
    halt_reason = None
    # Commands persist across chunk boundaries because the observation lag (~6 steps at
    # 30 Hz) routinely reaches back past the start of the current chunk.
    history: List[np.ndarray] = [q14.astype(np.float64)]
    # Bound to the caller's record, not to fresh locals: every append below has to be
    # visible from the `finally` that writes the artifacts, including when this frame
    # is unwound by an interrupt. Collected unconditionally -- a few arrays against a
    # rollout that already holds the arms for hundreds of ms per chunk. `--video` is
    # the exception: it costs a server-side VAE decode, so it stays opt-in.
    lags, tracks = record["lags"], record["track"]
    pred_rel, cmd_abs = record["pred_rel"], record["cmd_abs"]
    meas_abs, meas_steps = record["meas_abs"], record["meas_steps"]
    bounds, videos = record["bounds"], record["videos"]

    while chunk_i < args.max_chunks and not stopper.stop:
        t_infer = time.perf_counter()
        got = va.infer(first_obs, want_video=args.video)   # (16 or 14, F, N)
        action, imagined = got if args.video else (got, None)
        # The round trip is three unrelated costs, and the server reports the split:
        #   vgen -- denoising the video latents   (num_inference_steps)
        #   act  -- denoising the action chunk    (action_num_inference_steps)
        #   vid  -- BOTH VAE decodes, only with --video: the imagined latents here,
        #           and the real observation latents inside va.advance
        # vgen runs even without --video, because the action loop attends to the latents
        # it produces. `other_ms` is whatever the round trip cost beyond the three --
        # transport, msgpack, the obs encode on chunk 0 -- kept rather than dropped, so
        # the parts always add up to the wall clock the arms actually waited.
        # The imagined decode; the real-stream one is added after va.advance below.
        video_ms = va.last_video_decode_ms
        vgen_ms = va.last_video_gen_ms
        act_ms = va.last_action_gen_ms
        infer_ms = (time.perf_counter() - t_infer) * 1e3 - video_ms
        other_ms = infer_ms - vgen_ms - act_ms

        if action.ndim != 3 or action.shape[0] != conv.action_dim:
            raise RuntimeError(f"unexpected action shape {action.shape} for a "
                               f"{conv.action_space} checkpoint")
        n_per_frame = action.shape[2]
        if n_per_frame % 4 != 0:
            raise RuntimeError(f"action_per_frame {n_per_frame} is not a multiple of 4")
        steps_per_keyframe = n_per_frame // 4
        record["keyframe_nominal_ms"] = steps_per_keyframe * period * 1e3

        # Latent frame 0 of the first chunk is the conditioning frame rebuilt from the
        # initial observation; its actions are not a command. Under served_metadata's
        # frame0_action 'train_pad' they are training's pad -- raw zeros: the anchor
        # translation for an EEF checkpoint, all-zero joints for a joint one, closed
        # grippers for both; under 'denoised' they are the model's own sample of that
        # pad. A server without the field used normalized zeros, which decode to
        # mid-range poses ~18 cm from the anchor. `state` below still gets the full
        # array.
        start_idx = 1 if first else 0

        t_ik = time.perf_counter()
        before_unsafe = conv.stats["unsafe_jumps"]
        try:
            cmds = conv.convert_chunk(action[:, start_idx:, :])
        except ValueError as exc:
            # A malformed action (degenerate quaternion, wrong shape) should stop the
            # rollout with the arms holding, not propagate out and take the process
            # with it -- an uncaught exception here disconnects from raiden, which
            # e-stops and homes. Safe, but it discards the diagnostic and the summary.
            halt_reason = f"could not convert chunk {chunk_i}: {exc}"
            break
        ik_ms = (time.perf_counter() - t_ik) * 1e3

        if conv.stats["unsafe_jumps"] > before_unsafe:
            # An IK seed stalled and was rescued from a fallback seed, which switches
            # solution branch. Executing that step is a large slew that raiden's
            # _check_joint_delta would emergency_stop on.
            halt_reason = (f"IK branch switch of "
                           f"{conv.stats['max_fallback_jump_rad']:.3f} rad "
                           f"(> {conv.max_jump_rad}); holding instead of slewing")
            break

        # Bound the per-step change. This runs *after* the unsafe_jumps gate above,
        # which is load-bearing: a branch switch must be refused, never interpolated.
        cmds, slew = slew_limit(cmds, history[-1], args.max_step_delta)
        if slew["clipped_steps"]:
            # Say so the moment it happens, not only in the chunk summary. Clipping
            # means the arms are being sent something other than what the policy
            # asked for, and the policy cannot tell -- it is fed back its own
            # emitted actions, never measured state.
            joints = ", ".join(f"{j}x{n}" for j, n in slew["joints"].items())
            print(f"  SLEW CLIP: {slew['clipped_steps']}/{len(cmds)} steps limited to "
                  f"{args.max_step_delta} rad (largest request "
                  f"{slew['max_requested_step']:.3f}) on {joints}; trajectory is "
                  f"{slew['max_deviation']:.3f} rad behind what was asked for",
                  flush=True)
        if slew["max_deviation"] > args.max_slew_deviation:
            halt_reason = (
                f"slew limiter fell {slew['max_deviation']:.3f} rad behind the "
                f"requested trajectory (> {args.max_slew_deviation}). The policy is "
                "commanding motion faster than --max-step-delta allows, and it cannot "
                "see that the arms are not following.")
            break

        # Recorded after every gate above has passed, so the arrays hold what was
        # actually dispatched. `cmds` is post-slew, which is the point of plotting it:
        # predicted-vs-commanded is exactly the distortion IK and the limiter added.
        bounds.append(executed)
        pred_rel.append(conv.predicted_rel16(action[:, start_idx:, :]))
        cmd_abs.extend(joints14_to_eef16(c, conv.kin_left, conv.kin_right) for c in cmds)

        keyframes: List[Dict[str, np.ndarray]] = []
        keyframe_ts: List[float] = []
        deadline = time.perf_counter()
        max_track = 0.0

        for i, cmd in enumerate(cmds):
            if not args.dry_run:
                robot.put_action(cmd.astype(np.float32))
            history.append(np.asarray(cmd, dtype=np.float64))
            executed += 1
            # Per step, not per chunk: it is what finalize_record truncates the recorded
            # series to, and a halt or interrupt can land anywhere in the chunk.
            record["steps"] = executed
            deadline += period
            time.sleep(max(0.0, deadline - time.perf_counter()))

            if (i + 1) % steps_per_keyframe == 0:
                obs = robot.latest_obs
                if obs is None:
                    halt_reason = "no observation from raiden yet"
                    break
                keyframes.append(build_obs(obs))
                # raiden stamps each observation with its reference camera's capture
                # time. Training placed keyframes exactly steps_per_keyframe frames apart;
                # these come from the latest --obs-hz poll, so their real spacing is what
                # the world model is actually being shown.
                keyframe_ts.append(float(getattr(obs, "timestamp", 0.0)))
                # Meaningless under --dry-run: nothing was commanded, so the arm sits
                # still while the commands advance and this would always trip.
                if not args.dry_run:
                    queued = len(getattr(robot, "_action_queue", ()))
                    if queued > args.max_backlog:
                        halt_reason = (
                            f"action backlog {queued} > {args.max_backlog}: the "
                            f"dispatcher is not keeping up, so the arms are "
                            f"{queued / args.control_hz:.2f}s behind the policy and "
                            "every timing figure is meaningless. Raise "
                            "--dispatch-mult.")
                        break
                    # Compare against the command the measurement actually reflects,
                    # not the one just issued: at --obs-hz 10 the observation is up to
                    # 100 ms stale, which at 30 Hz is 0.3+ rad of command travel and
                    # would read as a tracking failure on a healthy arm. Both arms --
                    # comparing only `[:6]` would monitor the left and ignore the right.
                    window = np.asarray(history[-(args.max_obs_lag + 1):])
                    track, lag, _ = tracking_error(
                        window, len(window) - 1, measured_joints(obs),
                        args.max_obs_lag)
                    max_track = max(max_track, track)
                    # Every sample, dwells included: a flat error curve leaves the lag
                    # unidentifiable but the error itself valid (tracking_error).
                    tracks.append(track)
                    if lag is not None:
                        lags.append(lag)
                        # Recorded only when the lag is identifiable. `tracking_error`
                        # returns None exactly when the error curve is flat across the
                        # whole window -- the arm is dwelling, so the sample says
                        # nothing about where in the trajectory it is, and plotting it
                        # would draw a meaningless plateau. Not lag-compensated: the
                        # sample is placed at the step it was taken, so a healthy arm
                        # reads ~5-6 steps behind the command. The plot subtitle says so.
                        meas_abs.append(joints14_to_eef16(
                            measured_joints(obs), conv.kin_left, conv.kin_right))
                        meas_steps.append(executed)
                    if track > args.max_tracking_error:
                        halt_reason = (f"tracking error {track:.3f} rad exceeds "
                                       f"{args.max_tracking_error} even at best lag "
                                       f"{lag} -- the arm is not following; collision "
                                       "or torque limit?")
                        break

        # Within the chunk only: the gap across a chunk boundary spans the inference
        # hold, and says nothing about sampling. A zero gap means the poll had not
        # delivered a new frame, so the world model was shown the same instant twice.
        # A missing timestamp (0.0) would read as exactly that, so it is skipped.
        if len(keyframe_ts) > 1 and min(keyframe_ts) > 0:
            gaps = np.diff(keyframe_ts) * 1e3
            record["keyframe_gaps_ms"].extend(gaps.tolist())
            repeats = int((gaps <= 0).sum())
            if repeats:
                print(f"  WARNING: {repeats} keyframe(s) repeated the previous frame -- "
                      f"--obs-hz {args.obs_hz:g} did not deliver a new observation "
                      f"within {steps_per_keyframe * period * 1e3:.0f} ms. Raise it.",
                      flush=True)

        if halt_reason is not None:
            break

        # A ragged keyframe list desyncs the server's streaming VAE, so only advance
        # the world model when the chunk completed intact.
        expected = (action.shape[1] - start_idx) * 4
        if len(keyframes) != expected:
            halt_reason = (f"got {len(keyframes)} keyframes, expected {expected}; "
                           "skipping compute_kv_cache to avoid desyncing the VAE")
            break
        # Timed, because it runs in the same hold as infer and IK: it encodes the
        # keyframes and pushes two transformer forwards, and with --video it also
        # advances the decoder's real stream. It was previously omitted from the duty
        # cycle entirely, which overstated how much of the wall clock the arms spend
        # moving -- by more, not less, the more expensive the observation path got.
        t_adv = time.perf_counter()
        va.advance(keyframes, action)
        # Minus the real-stream decode the server does in here when video is on: that
        # belongs to `vid`, not to advancing the world model. Subtracting it is what
        # makes `adv` mean the same thing in a --video run as in a normal one, so the
        # two are comparable.
        advance_ms = (time.perf_counter() - t_adv) * 1e3 - va.last_real_decode_ms
        # Two decodes of the same latent count through the same decoder, so they should
        # be within noise of each other; a lopsided split means something is wrong.
        video_ms += va.last_real_decode_ms
        if args.video:
            # `keyframes` is already the [{cam_key: RGB}] shape save_imagination wants.
            videos.append((imagined, keyframes))

        # Inference, IK and advance all run while the arms hold position -- the stages
        # are strictly sequential, because the next infer needs a world model advanced
        # by observations captured during this chunk. So every one of them belongs in
        # the duty cycle; counting only inference overstates it. Video decoding is in
        # here too when it is on, which is exactly why --video is not a free diagnostic:
        # the duty cycle it reports is the duty cycle of a run that is decoding video.
        exec_s = len(cmds) * period
        hold_s = (infer_ms + ik_ms + advance_ms + video_ms) / 1e3
        duty = exec_s / max(1e-6, exec_s + hold_s)
        lag_s = f"{np.median(lags):.0f}" if lags else "-"
        vid_s = f"  vid {video_ms:5.0f}ms" if args.video else ""
        oth_s = f"  oth {other_ms:5.0f}ms" if abs(other_ms) > 50 else ""
        print(f"[chunk {chunk_i:3d}] vgen {vgen_ms:6.0f}ms  act {act_ms:6.0f}ms  "
              f"ik {ik_ms:6.1f}ms  adv {advance_ms:6.0f}ms{vid_s}{oth_s}  "
              f"{len(cmds):2d} steps  {len(keyframes)} keyframes  "
              f"duty {duty * 100:4.1f}%  track {max_track:.3f} rad  "
              f"lag {lag_s}  slew {slew['clipped_steps']}/{len(cmds)}"
              f"@{slew['max_deviation']:.3f}", flush=True)
        if ik_ms > exec_s * 1e3:
            # A failed solve costs ~196 ms: ik_robust retries every fallback seed, and
            # no seed can reach a target that is out of reach. Measured at an anchor
            # offset of 14 mm, ~12% of steps miss the 1e-4 m tolerance and IK goes from
            # 0.87 to 48.6 ms/step. About 8 such misses per 96-solve chunk is all the
            # budget allows before IK alone outlasts the motion it is planning.
            print(f"  WARNING: IK took {ik_ms:.0f}ms for {exec_s * 1e3:.0f}ms of "
                  "motion -- targets are drifting out of reach. Loosen the converter's "
                  "IK tolerance before this compounds.", flush=True)

        first = False
        chunk_i += 1
        # Mirrored every chunk rather than only on return, so an interrupt leaves the
        # record reporting what had actually been executed up to that point.
        record["chunks"], record["steps"] = chunk_i, executed

    record["chunks"], record["steps"] = chunk_i, executed
    record["halt_reason"] = halt_reason
    return record


def _series_error(a: np.ndarray, b: np.ndarray, at: np.ndarray = None) -> dict:
    """Per-arm translation (mm) and rotation (deg) error between two EEF series."""
    out = {}
    for arm, (xyz, quat, grip) in ARMS.items():
        lhs = a if at is None else a[at]
        terr = np.linalg.norm(lhs[:, xyz] - b[:, xyz], axis=1) * 1000
        rerr = rotation_error_deg(lhs[:, quat], b[:, quat])
        out[arm] = {
            "trans_mm": np.percentile(terr, [50, 99]).tolist() + [float(terr.max())],
            "rot_deg": np.percentile(rerr, [50, 99]).tolist() + [float(rerr.max())],
            "grip_mae": float(np.abs(lhs[:, grip] - b[:, grip]).mean()),
        }
    return out


def load_results(out_dir: str) -> list:
    """Existing results.json for this checkpoint, or [] if there is none.

    Read so a new session appends to it. Written back whole after every rollout, which
    is what keeps the file consistent with the plots when a session is interrupted.

    A results.json that cannot be parsed is set aside rather than discarded: it is the
    record of rollouts that already happened and cannot be produced again.
    """
    path = os.path.join(out_dir, "results.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            existing = json.load(f)
        if isinstance(existing, list):
            return existing
        raise ValueError(f"expected a list, got {type(existing).__name__}")
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        backup = path + ".corrupt"
        print(f"[out] {path} is unreadable ({exc}); moving it to {backup} and "
              f"starting a new one. The plots beside it are untouched.")
        with contextlib.suppress(OSError):
            os.replace(path, backup)
        return []


def write_artifacts(summary: dict, args) -> dict:
    """Trajectory plot, imagination video and results.json for one rollout.

    Called from a `finally`, so a rollout stopped by a safeguard or by Ctrl-C still
    leaves a record -- those are the runs whose plot is worth the most, and they are
    exactly the ones a write-on-success path would discard.
    """
    pred, cmd = summary["predicted"], summary["commanded"]
    meas, msteps = summary["measured"], np.asarray(summary["measured_steps"], dtype=int)
    metrics = {}
    # Unconditional: results.json is written next to this even when the rollout halted
    # before executing anything, and that empty-but-present record is itself the
    # finding (halt_reason says why nothing ran).
    os.makedirs(args.out, exist_ok=True)
    if not len(pred):
        print("  [artifacts] nothing executed; results.json only, no plot")
    else:
        tag = f"ep{summary['rollout']}"
        # Filenames carry the inference settings; the plot title stays readable.
        stem = artifact_stem(summary["rollout"], args.param_tag)
        series = [
            {"name": "predicted", "data": pred, "style": {"ls": "--", "lw": 1.4}},
            {"name": ("commanded (IK+slew)" if args.action_space == "eef"
                      else "commanded (clip+slew)"),
             "data": cmd, "style": {"ls": "-", "lw": 1.6}},
        ]
        if len(meas):
            series.append({"name": "measured", "data": meas, "t": msteps})
        halted = summary["halt_reason"]
        plot_trajectories(
            series, summary["bounds"], os.path.join(args.out, f"{stem}_traj.png"),
            f"{tag} (closed loop)" + ("  —  HALTED" if halted else ""),
            subtitle="measured: keyframe rate, dwell samples dropped, NOT "
                     "lag-compensated (~5-6 steps of pipeline lag expected)")
        metrics["predicted_vs_commanded"] = _series_error(pred, cmd)
        if len(meas):
            # Time-aligned, NOT lag-compensated, like the plot: mostly how far the arm
            # moves during its ~5-6 steps of lag. Tracking accuracy is `tracking_rad`.
            metrics["commanded_vs_measured"] = _series_error(cmd, meas, at=msteps)
        if args.video and summary["videos"]:
            save_imagination(summary["videos"],
                             os.path.join(args.out, f"{stem}_imagined.mp4"),
                             labels=args.labels, real_caption="live observation")
        print(f"  [artifacts] wrote {stem}_* to {args.out}/")

    return {
        "rollout": summary["rollout"], "chunks": summary["chunks"],
        "steps": summary["steps"],
        "halt_reason": summary["halt_reason"], "stats": summary["stats"],
        "lag_p50": pct(summary["lags"], 50) if summary["lags"] else None,
        "lag_p90": pct(summary["lags"], 90) if summary["lags"] else None,
        "n_lag_samples": len(summary["lags"]),
        "n_measured_samples": int(len(meas)),
        # [p50, p99, max] of the lag-compensated error --max-tracking-error halts on:
        # the worst arm joint, rad, at the best of the last --max-obs-lag commands.
        "tracking_rad": ([pct(summary["track"], 50), pct(summary["track"], 99),
                          float(np.max(summary["track"]))]
                         if summary["track"] else None),
        "keyframe_gap_ms": _keyframe_gaps(summary),
        "arms": metrics,
        # results.json accumulates every session against this checkpoint, and the
        # filename tag carries only the step counts.
        "served": args.served_settings,
    }


def _keyframe_gaps(summary: dict):
    """How far apart the keyframes the world model saw really were, vs training.

    Training sampled them exactly `action_per_frame/4` frames apart. Here each is the
    latest --obs-hz poll, up to one poll interval old, so the spacing jitters around
    the nominal value by up to that interval either way, and 0 means a repeated frame.
    """
    gaps = summary["keyframe_gaps_ms"]
    if not gaps:
        return None
    return {"nominal": summary["keyframe_nominal_ms"],
            "min": float(np.min(gaps)), "p50": pct(gaps, 50), "max": float(np.max(gaps)),
            "repeats": int(sum(g <= 0 for g in gaps)), "n": len(gaps)}


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
                    help="Observation stream rate. A keyframe is taken every "
                         "action_per_frame/4 control steps (200 ms for cover_blocks, "
                         "133 ms for place_cube) from the latest polled observation, so "
                         "it can be up to one poll interval older than the frame "
                         "training put there -- 100 ms, 3 steps, at 10 Hz. Higher "
                         "tightens that but costs bandwidth (~2.8MB per frame at "
                         "480x640 x3 cameras, uncompressed). The measured spacing is "
                         "printed and saved as keyframe_gap_ms.")
    ap.add_argument("--max-chunks", type=int, default=40,
                    help="Chunk budget per episode (~1.6s of motion each).")
    ap.add_argument("--max-jump-rad", type=float, default=0.2,
                    help="IK branch-switch threshold; above this the chunk is refused. "
                         "EEF checkpoints only: a joint checkpoint has no IK.")
    ap.add_argument("--max-tracking-error", type=float, default=0.45,
                    help="Halt if measured joints diverge from commanded by more than "
                         "this, after lag compensation. The policy cannot see this "
                         "divergence itself. 0.45 sits just under the 0.5 the replay "
                         "validated, so this client stops before rd serve e-stops.")
    ap.add_argument("--dispatch-mult", type=float, default=4.0,
                    help="Run chiral's action dispatcher at this multiple of "
                         "--control-hz. Its loop sleeps a fixed interval rather than "
                         "tracking an absolute deadline, so asking for exactly "
                         "--control-hz runs ~2%% slow and the queue grows without "
                         "bound.")
    ap.add_argument("--max-backlog", type=int, default=10,
                    help="Halt if this many actions are queued but unsent. Backlog is "
                         "latency the arms have not begun working through.")
    ap.add_argument("--max-step-delta", type=float, default=0.35,
                    help="Clip per-step arm-joint change to this, matching rd serve's "
                         "--max-joint-delta so a single command step can never on its "
                         "own exceed the server's e-stop threshold. Healthy converted "
                         "motion measures p99.99 = 0.083 rad and max 0.086, so this "
                         "sits ~4x above normal and binds only on gross anomalies; "
                         "lower it toward 0.15 for a tighter guard. Grippers are not "
                         "limited, and an IK branch switch is refused upstream rather "
                         "than interpolated through here.")
    ap.add_argument("--max-slew-deviation", type=float, default=0.3,
                    help="Halt if clipping has pushed the executed trajectory this "
                         "far from the requested one. Clipping rewrites what the arms "
                         "do, and the policy is fed back its own actions so it never "
                         "learns it is being overridden.")
    ap.add_argument("--max-obs-lag", type=int, default=24,
                    help="How many recent commands to search when matching a "
                         "measurement. Expect a true lag of ~5-6 steps at 30 Hz, of "
                         "which ~3 is the arm's own position-loop response.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the full loop but never send actions to the robot. Still "
                         "reads real observations, so it measures inference latency and "
                         "exercises the whole protocol with the arms untouched.")
    ap.add_argument("--config-name", default=None,
                    help="The CONFIG_NAME the policy server was launched with. The "
                         "server reports it, so this is only a check that stops the run "
                         "on a mismatch -- and the identity itself for a server too old "
                         "to report one. It does not change what is evaluated.")
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint directory being served (parent of transformer/). "
                         "Checked against what the server reports it loaded; needed "
                         "only when the serving directory holds real files, not links.")
    ap.add_argument("--out", default=None,
                    help="Directory for the rollout's trajectory plot, imagination "
                         "video and results.json. Defaults to the checkpoint's "
                         "position mirrored under eval_out/, plus closed_loop/.")
    ap.add_argument("--no-artifacts", dest="save_artifacts", action="store_false",
                    help="Skip the trajectory plot and results.json. They are written "
                         "by default: the plot overlays the model's prediction, what "
                         "IK and the slew limiter actually commanded, and what the "
                         "arms measurably did, so a divergence can be attributed to "
                         "the conversion pipeline or to the hardware rather than just "
                         "observed. A real-robot rollout is the most expensive run in "
                         "this project and the least repeatable, so it defaults to "
                         "leaving a record -- including when a safeguard halts it or "
                         "you Ctrl-C out. Costs a few arrays and one matplotlib call "
                         "after the arms have stopped; nothing on the control path.")
    ap.add_argument("--video", action="store_true",
                    help="Also dump the imagined world-model canvas beside the live "
                         "camera view. OFF by default "
                         "because it is NOT free here: it adds a server-side VAE "
                         "decode to every infer, and the loop is strictly sequential "
                         "-- the arms hold position throughout. That lengthens the "
                         "inter-chunk hold and so changes the duty cycle the run is "
                         "measuring. Prefer pairing it with --dry-run.")
    ap.add_argument("--no-labels", dest="labels", action="store_false",
                    help="Render the video without the 'imagined'/'live observation' "
                         "captions and the chunk/frame counters (on by default).")
    args = ap.parse_args()
    if args.video and not args.save_artifacts:
        raise SystemExit("--video and --no-artifacts contradict each other: the video "
                         "is one of the artifacts. Drop one.")

    from chiral import PolicyClient  # noqa: E402  (raiden venv only)

    va = VAClient(host=args.va_host, port=args.va_port)

    # Resolved before the arms are touched, from what the policy server says it loaded:
    # a --config-name or --checkpoint that disagrees with it should stop the run here,
    # not after a rollout has been filed under the wrong model.
    served = resolve_served(va.server_metadata, args.config_name, args.checkpoint)
    print(f"[va] serving {served['config_name']} from "
          f"{served['checkpoint'] or '<unlinked serving directory>'}")
    args.param_tag = served['param_tag']
    args.served_settings = served['settings']
    print(f"[va] settings {served['settings']}")
    # The checkpoint decides what its actions are, so the converter follows the server.
    args.action_space = served['action_space']
    conv = make_converter(args.action_space, args.i2rt_root,
                          max_jump_rad=args.max_jump_rad)
    print(f"[va] action space: {args.action_space} "
          f"({'IK on EEF poses' if args.action_space == 'eef' else 'joints, no IK'})")
    if args.out is None:
        args.out = eval_out_dir('closed_loop', served['checkpoint'])
    if args.save_artifacts:
        print(f"writing artifacts to {args.out}/ as episode_*_{args.param_tag}_*")

    stopper = Stopper()

    if args.dry_run:
        print("[dry-run] actions will NOT be sent to the robot")

    # One connection for the whole session: disconnecting triggers raiden's
    # emergency_stop(), which homes the arms and calls os._exit(0) on the server.
    with PolicyClient(args.robot_uri) as robot:
        meta = robot.get_metadata()
        print(f"[robot] connected: cameras={meta.get('cameras')}, "
              f"control_hz={meta.get('control_hz')}, "
              f"max_joint_delta={meta.get('max_joint_delta')}")
        # max_joint_delta=None: this client has no preflight to compare against --
        # it converts and dispatches one chunk at a time. The server's value is
        # reported so the e-stop threshold is visible before anything moves.
        fails = check_server_metadata(meta, args.control_hz, None, BIMANUAL_DOF)
        if fails:
            raise SystemExit("\n".join("ERROR: " + f for f in fails))

        # Rollouts continue the checkpoint's existing numbering, and results.json is
        # extended rather than replaced, so a second session against the same
        # checkpoint adds to the record instead of erasing the first one's.
        first_rollout = next_rollout_index(args.out)
        results = load_results(args.out)
        if first_rollout:
            print(f"[out] {args.out} already holds rollouts 0..{first_rollout - 1}; "
                  f"this session starts at {first_rollout}")

        rollout = first_rollout
        print(f"\n=== rollout {rollout}: {args.prompt!r} ===")
        # Created here, not inside run_episode: a second Ctrl-C unwinds that frame
        # and everything in its locals with it. The record survives, so the
        # `finally` below always has whatever was collected before the interrupt.
        summary = new_record()
        summary["rollout"] = rollout
        interrupted = False
        try:
            run_episode(robot, va, conv, args.prompt, args, stopper, summary)
        except VAServerDied as exc:
            print(f"\n[fatal] policy server died and must be relaunched:\n{exc}")
            return 1
        except KeyboardInterrupt:
            # The second Ctrl-C. "Exit immediately" means stop commanding the arms
            # -- which stop_action_dispatch below does -- not throw away the
            # rollout. Writing a PNG touches no hardware.
            summary["halt_reason"] = "interrupted (second Ctrl-C)"
            interrupted = True
        finally:
            robot.stop_action_dispatch()
            if args.save_artifacts:
                with sigint_deferred("artifacts"):
                    finalize_record(summary, conv)
                    results.append(write_artifacts(summary, args))
                    os.makedirs(args.out, exist_ok=True)
                    with open(os.path.join(args.out, "results.json"), "w") as f:
                        json.dump(results, f, indent=2)
        if interrupted:
            return 1

        print(f"[rollout {rollout}] {summary['chunks']} chunks, "
              f"{summary['steps']} steps executed")
        if summary["halt_reason"]:
            print(f"[rollout {rollout}] HALTED: {summary['halt_reason']}")
        s = summary["stats"]
        if args.action_space == "eef":
            print(f"  IK: {s['ik_failures']} failures, {s['ik_fallbacks']} fallbacks, "
                  f"max residual {s['max_pos_residual_m'] * 1000:.2f}mm / "
                  f"{s['max_ori_residual_rad']:.1e}rad, "
                  f"gripper clipped {s['gripper_clipped']}")
        else:
            print(f"  joints: {s['joint_limit_clipped']}/{s['steps']} steps clipped to "
                  f"the joint limits (worst overshoot "
                  f"{s['max_joint_overshoot_rad']:.3f} rad), "
                  f"gripper clipped {s['gripper_clipped']}")
        lags = summary["lags"]
        if lags:
            print(f"  observation lag: p50={pct(lags, 50):.0f} p90={pct(lags, 90):.0f}"
                  f" steps from {len(lags)} identifiable samples "
                  f"(~5-6 expected at 30 Hz; far above means queueing)")
        if summary["track"]:
            t = summary["track"]
            print(f"  tracking error (lag-compensated): p50={pct(t, 50):.3f} "
                  f"p99={pct(t, 99):.3f} max={max(t):.3f} rad "
                  f"(halts above {args.max_tracking_error})")
        kf = _keyframe_gaps(summary)
        if kf:
            print(f"  keyframe spacing: min={kf['min']:.0f} p50={kf['p50']:.0f} "
                  f"max={kf['max']:.0f} ms vs {kf['nominal']:.0f} ms in training"
                  + (f", {kf['repeats']} repeated frame(s)" if kf["repeats"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
