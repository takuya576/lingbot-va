# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Observation decoding and command monitoring for a live raiden connection.

Shared by `script/run_yam_policy.py` and `script/replay_yam_episode.py` so the two
measure the same things the same way -- the replay exists to certify the path the
policy client then drives, which it cannot do if the monitors differ.

Deliberately numpy-only, like the rest of `yam_bridge`: it is imported in raiden's
venv, which has no torch. `chiral` types are duck-typed rather than imported, so this
module also loads in a plain lingbot-va environment for offline analysis.

Everything here exists because the policy is never shown the robot. `format_obs`'s
`observation.state` key is never read (`wan_va/wan_va_server.py:583`) -- the model is
fed back the chunk it just emitted -- so IK residual, silent joint clipping and
tracking error never reach it. Only the cameras close the loop. These monitors are
the only thing watching the difference between commanded and achieved.
"""
import numpy as np

from yam_bridge.action_convert import BIMANUAL_DOF, YAM_DOF
from yam_bridge.kinematics import ARM_DOF

# The 12 revolute joints, gripper channels excluded. `_check_joint_delta` applies its
# radian threshold to `[:6]` of each arm only, for the same reason: the gripper is a
# normalized [0, 1] channel and a radian limit on it is meaningless.
ARM_JOINTS = np.r_[0:ARM_DOF, YAM_DOF:YAM_DOF + ARM_DOF]

PROPRIO_KEYS = ("follower_l_joint_pos", "follower_r_joint_pos")

# raiden camera name -> LingBot-VA observation key. Identical to `DEFAULT_CAMERA_MAP`
# in raiden/lerobot_export.py:59-63, which is the mapping the training dataset was
# built with. Order matters: index 0 is the full-resolution camera in the server's
# T-shape mosaic.
CAMERA_MAP = {
    "scene_camera": "observation.images.cam_high",
    "left_wrist_camera": "observation.images.cam_left_wrist",
    "right_wrist_camera": "observation.images.cam_right_wrist",
}

# Roughly what a 1 GbE link sustains, minus headroom. Observations and actions share
# one socket and raiden's handler is sequential per connection, so an observation on
# the wire delays the actions queued behind it.
LINK_MB_S = 110.0

# How much the best-lag error curve must vary across the search window before the
# position of its minimum means anything. Well above the ~0.02 rad noise floor the
# measured tracking error sits at, and well below the motion of any real move.
LAG_IDENTIFIABLE_RAD = 0.05


def pct(x, p: float) -> float:
    return float(np.percentile(np.asarray(x).reshape(-1), p))


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


def build_obs(observation) -> dict:
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


def check_server_metadata(meta: dict, control_hz: float, max_joint_delta, dof: int) -> list:
    """Compare `rd serve`'s configuration against this client's. Returns failures.

    Everything here is a mismatch nothing downstream would notice. `apply_action` is
    fire-and-forget with no response (chiral/protocol.py:14-15), so a client sending
    at 30 Hz to a server interpolating for 10 Hz gets no error -- only motion the arms
    cannot follow. Likewise a client that gates its own trajectory on a joint-delta
    threshold looser than the server's passes its own preflight and is then e-stopped
    mid-run by `_check_joint_delta`.

    Pass `max_joint_delta=None` from a client that does not gate on it -- the live
    policy client converts one chunk at a time and dispatches it immediately, so it
    has no trajectory to check ahead of time and nothing to compare. The server's
    value is still printed, because it is the threshold that will e-stop the run.

    `control_hz` and `max_joint_delta` were added to raiden's metadata for this check;
    an older server omits them, which is reported rather than treated as agreement.
    """
    fails = []
    shape = list(meta.get("action_shape", []))
    if meta.get("action_type") != "joint" or shape != [dof]:
        fails.append(
            f"rd serve is in {meta.get('action_type')!r} mode with action_shape "
            f"{shape}; this client sends {dof}-D joint commands. Relaunch with "
            "--action-type joint.")

    server_hz = meta.get("control_hz")
    if server_hz is None:
        print("  NOTE: this rd serve predates control_hz in metadata -- the rate "
              "cannot be verified. Check it matches --control-hz by hand.")
    elif abs(float(server_hz) - control_hz) > 1e-6:
        fails.append(
            f"rd serve --control-hz is {server_hz:g} but this client paces at "
            f"{control_hz:g}. Nothing detects this at run time: actions are "
            "fire-and-forget, so the arms would simply be driven at the wrong rate.")

    server_delta = meta.get("max_joint_delta")
    if server_delta is None:
        print("  NOTE: this rd serve predates max_joint_delta in metadata.")
    elif max_joint_delta is None:
        print(f"  NOTE: rd serve will emergency_stop if a commanded joint diverges "
              f"from measured by more than {float(server_delta):g} rad.")
    elif float(server_delta) < max_joint_delta - 1e-9:
        fails.append(
            f"rd serve --max-joint-delta is {server_delta:g}, tighter than this "
            f"client's {max_joint_delta:g}. Preflight would pass a trajectory the "
            "server then e-stops on. Match them, or loosen the server.")
    elif float(server_delta) > max_joint_delta + 1e-9:
        print(f"  NOTE: rd serve --max-joint-delta {server_delta:g} is looser than "
              f"this client's {max_joint_delta:g}; the client gates first.")
    return fails


def slew_limit(cmds: np.ndarray, last_cmd: np.ndarray, max_step: float) -> tuple:
    """Bound the per-step change of a command stream. Returns `(limited, stats)`.

    Only the 12 arm joints are limited. The gripper channels are normalized [0, 1] and
    a radian-per-step bound on them is meaningless -- the same reason
    `_check_joint_delta` only inspects `[:6]` of each arm.

    **Do not route an IK branch switch through this.** A branch switch reaches the same
    end-effector pose from a different arm configuration; interpolating between the two
    in joint space traverses poses that were never asked for, so spreading it over more
    steps makes it slower, not safer. The caller must gate on `unsafe_jumps` first and
    hold position instead.

    What this *is* for is an ordinary step larger than the arms can track, where
    neighbouring IK solutions lie on the same branch and interpolation is sane.

    The cost is that clipping silently rewrites the trajectory, and the policy cannot
    see that: it is fed back the chunk it emitted, not measured state
    (`wan_va_server.py:583`), so a persistently binding limiter means the arms execute
    something the policy never asked for and never finds out. `stats["max_deviation"]`
    is how far the limited stream has fallen behind the requested one -- the caller
    should treat sustained growth there as a halt condition, not a nuisance.
    """
    cmds = np.asarray(cmds, dtype=np.float64)
    limited = cmds.copy()
    prev = np.asarray(last_cmd, dtype=np.float64).copy()
    clipped = 0
    # Which joints bound, and by how much. Naming them is most of the diagnostic
    # value: a limiter firing on one wrist joint is a different problem from one
    # firing across a whole arm, and the aggregate count cannot tell them apart.
    hits = np.zeros(len(ARM_JOINTS), dtype=int)
    worst = 0.0
    for i in range(len(limited)):
        delta = limited[i, ARM_JOINTS] - prev[ARM_JOINTS]
        over = np.abs(delta) > max_step
        if over.any():
            clipped += 1
            hits += over
            worst = max(worst, float(np.abs(delta).max()))
            limited[i, ARM_JOINTS] = (prev[ARM_JOINTS]
                                      + np.clip(delta, -max_step, max_step))
        prev = limited[i]
    deviation = np.abs(limited[:, ARM_JOINTS] - cmds[:, ARM_JOINTS]).max(axis=1)
    names = [f"{'LR'[k // ARM_DOF]}{k % ARM_DOF}" for k in range(len(ARM_JOINTS))]
    return limited, {
        "clipped_steps": clipped,
        "max_deviation": float(deviation.max()) if len(deviation) else 0.0,
        "final_deviation": float(deviation[-1]) if len(deviation) else 0.0,
        "max_requested_step": worst,
        "joints": {names[k]: int(n) for k, n in enumerate(hits) if n},
    }


def tracking_error(cmds: np.ndarray, i: int, measured: np.ndarray,
                   max_lag: int) -> tuple:
    """Smallest |measured - cmd| over the last `max_lag` commands, and which one won.

    Returns `(error, lag_or_None, uncompensated_error)`.

    A client-side sample cannot be compared against the command just dispatched. The
    observation is stale by the poll interval, plus transport, plus the camera capture
    time the proprios are interpolated onto (`_make_obs` stamps the observation with
    the reference camera's arrival time, raiden/server.py:1139-1141). At 30 Hz the
    command can advance 0.3+ rad in that window, so comparing against `cmds[i]` reports
    latency as if it were tracking failure.

    Searching over recent commands separates the two: the minimum is the real tracking
    error, and the lag at which it occurs is the pipeline latency -- the same best-lag
    idea `script/eval_yam_openloop.py` uses to validate its frame indexing. Large error
    at *every* lag is a genuine tracking failure, which this still catches.

    Expect ~5-6 steps of lag at 30 Hz, not 1-2. Measured from the dataset itself
    (`action` vs `observation.state` at the same indices, so no network and no
    observation pipeline at all), the arm lags its own commands by **3 steps**, scaling
    with the position gains: kp=80 joints lag 3, kp=40 lags 2, and the deliberately soft
    kp=10 wrist joints lag 5. On top of that `_smooth_command` spreads the move across
    `_SMOOTH_DUTY` of the period and only reaches the target at the end (+0.9 step, by
    design), the client's `latest_obs` is up to one poll interval old, and `_make_obs`
    interpolates proprios onto the newest camera frame. Only about two of those steps
    are pipeline, and the rest was equally present in the training data. A figure far
    above this range means queueing, not mechanics.

    Note the result is still an upper bound on what `_check_joint_delta` sees: the
    server compares against its own proprio buffer, refreshed at 100 Hz
    (server.py:1491-1499), so its measurement is ~10 ms old rather than ~200 ms.
    """
    lo = max(0, i - max_lag)
    errs = np.abs(cmds[lo:i + 1, ARM_JOINTS] - measured[ARM_JOINTS]).max(axis=1)
    best = int(np.argmin(errs))
    # The lag is only identifiable while the arm is moving. Across a dwell every
    # command in the window matches the measurement equally well, argmin lands
    # arbitrarily, and averaging those in corrupts the estimate -- cover_blocks is
    # stationary at least half the time (command step delta p50 is 0.0000 rad). Report
    # the lag only when the error curve has a trough; the error itself stays valid
    # either way, since a flat curve means every candidate agrees on it.
    spread = float(errs.max() - errs.min())
    lag = i - (lo + best) if spread > LAG_IDENTIFIABLE_RAD else None
    return float(errs[best]), lag, float(errs[-1])


def report_bandwidth(observation, obs_hz: float, need_images: bool) -> None:
    """Print the observation payload cost, and warn when polling will not fit.

    chiral sends observations uncompressed (`compression=None`, max_size=None), so the
    wire cost is just the array bytes. At 480x640x3 across three cameras, polling at
    30 Hz is 83 MB/s -- most of a 1 GbE link, and every millisecond an observation
    occupies the socket is a millisecond an action is not being dispatched.

    `need_images` changes the advice, not the arithmetic: a replay reads only proprios
    and can shrink the images away, while the policy needs them at the resolution it
    was trained on and must buy headroom by polling less often instead.
    """
    per_obs = sum(c.image.nbytes + (c.depth.nbytes if c.depth is not None else 0)
                  for c in observation.cameras)
    rate = per_obs * obs_hz / 1e6
    shapes = ", ".join(f"{c.name}={c.image.shape}" for c in observation.cameras)
    print(f"[obs] {per_obs / 1e6:.2f} MB/obs x {obs_hz:g} Hz = {rate:.1f} MB/s "
          f"({shapes})")
    if rate > LINK_MB_S / 3:
        fix = ("lower --obs-hz; the policy needs these images at training resolution, "
               "so the headroom has to come from the poll rate"
               if need_images else
               "relaunch with `rd serve --resize-images 64x64 --no-depth` -- this run "
               "reads only proprios")
        print(f"  WARNING: ~{rate / LINK_MB_S * 100:.0f}% of a 1 GbE link, sharing one "
              f"socket with action dispatch. Expect dispatch lateness and backlog "
              f"growth.\n  {fix}.")
