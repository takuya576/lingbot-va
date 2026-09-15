#!/usr/bin/env python3
"""Offline validation of the LingBot-VA -> YAM joint-action conversion.

Runs the full deployment chain against recorded `cover_blocks` episodes, where ground
truth exists, so the conversion can be verified before it ever drives real arms::

    joints14 --FK--> eef16 --relativize(anchor=frame 0)--> [quantize] --converter--> joints14'

This matters more than usual here because the policy is never told what the robot
actually did: `wan_va_server.py:583` feeds back the action chunk the model just
emitted, not measured state, so IK residual, joint clipping and tracking error are
invisible to it at run time. Offline is the only place they are observable.

Run in the raiden venv (mujoco + mink + scipy, no torch needed)::

    /data2/takuya/raiden/.venv/bin/python script/validate_yam_action_convert.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yam_bridge.action_convert import (  # noqa: E402
    BIMANUAL_DOF, DEFAULT_I2RT_ROOT, YAM_DOF, YamActionConverter, relativize_action16,
)
from yam_bridge.kinematics import ARM_DOF  # noqa: E402

ARM_JOINTS = np.r_[0:ARM_DOF, YAM_DOF:YAM_DOF + ARM_DOF]
GRIPPERS = [ARM_DOF, YAM_DOF + ARM_DOF]


def read_column(path: str, column: str) -> np.ndarray:
    """Read one FixedSizeList column of a LeRobot parquet as (N, D) float64."""
    table = pq.read_table(path, columns=[column])
    return np.stack(table.column(column).to_numpy(zero_copy_only=False)).astype(np.float64)


def quantize_roundtrip(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Apply the exact normalize -> clip -> denormalize the training pipeline uses.

    Forward is `lerobot_latent_dataset.py:279-281`, inverse is
    `wan_va_server.py:246-249`. Running both back to back measures what the q01/q99 fit
    costs in joint space: anything the quantiles cannot represent is clipped away here
    and is therefore unreachable by the model no matter how well it trains.
    """
    span = q99 - q01 + 1e-6
    normalized = np.clip((actions - q01) / span * 2.0 - 1.0, -1.5, 1.5)
    return (normalized + 1.0) / 2.0 * span + q01


def load_norm_stat(dataset_root: str) -> tuple:
    """Load the 16 used channels of q01/q99 from the converter's norm_stat.json.

    Read from the dataset rather than `va_yam_lora_train_cfg` on purpose: the config
    lives under `wan_va/`, whose `__init__` imports the whole training stack. The
    config's values are pasted from this file, so they are the same numbers.
    """
    with open(os.path.join(dataset_root, "norm_stat.json")) as f:
        stat = json.load(f)
    # Canonical 30-dim -> the 16 used channels, matching
    # `va_robotwin_cfg.used_action_channel_ids = [0..6, 28, 7..13, 29]`.
    ids = list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
    q01 = np.asarray(stat["q01"], dtype=np.float64)[ids]
    q99 = np.asarray(stat["q99"], dtype=np.float64)[ids]
    return q01, q99


def summarize(name: str, err: np.ndarray) -> dict:
    """Percentile summary of a per-step, per-joint error array (N, J) in radians."""
    flat = err.reshape(-1)
    return {
        "name": name,
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
    }


def run_episode(joints14: np.ndarray, eef16: np.ndarray, conv: YamActionConverter,
                quant: tuple = None, zero_seed: bool = False) -> dict:
    """Run one episode through the chain and compare against ground-truth joints."""
    relative = relativize_action16(eef16)
    if quant is not None:
        relative = quantize_roundtrip(relative, *quant)

    conv.reset_stats()
    conv.latch_anchor(joints14[0])

    t0 = time.perf_counter()
    if zero_seed:
        # Defeat seed carry-over to expose IK branch instability: re-latch the anchor
        # before every step so each solve starts from the rollout's initial pose.
        out = []
        for step in relative:
            conv.latch_anchor(joints14[0])
            out.append(conv.convert_step(step))
        joints_out = np.stack(out)
    else:
        joints_out = conv.convert_sequence(relative)
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    # Continuity, not agreement with the demonstration, is the safety-relevant
    # quantity: `_check_joint_delta` compares consecutive commands against measured
    # state and e-stops the process on a jump (`raiden/server.py:944-980`). Measured
    # as *excess* over the demonstration's own step size, since the demo is by
    # definition a trajectory the arm executed safely.
    step_ik = np.abs(np.diff(joints_out[:, ARM_JOINTS], axis=0)).max(axis=1)
    step_true = np.abs(np.diff(joints14[:, ARM_JOINTS], axis=0)).max(axis=1)

    return {
        "joints_out": joints_out,
        "arm_err": np.abs(joints_out[:, ARM_JOINTS] - joints14[:, ARM_JOINTS]),
        "grip_err": np.abs(joints_out[:, GRIPPERS] - joints14[:, GRIPPERS]),
        "excess_step": step_ik - step_true,
        "ms_per_step": elapsed_ms / max(len(relative), 1),
        "stats": dict(conv.stats),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Datasets/yam-real-world-lerobot/cover_blocks",
                    help="Raw joint-space LeRobot dataset (14-dim ground truth).")
    ap.add_argument("--dst", default="Datasets/yam-real-world-lingbot-va/cover_blocks",
                    help="Converted EEF dataset (16-dim) + norm_stat.json.")
    ap.add_argument("--i2rt-root", default=DEFAULT_I2RT_ROOT)
    ap.add_argument("--episodes", type=int, default=8,
                    help="How many episodes to validate (0 = all).")
    args = ap.parse_args()

    q01, q99 = load_norm_stat(args.dst)
    conv = YamActionConverter(args.i2rt_root)

    src_dir = os.path.join(args.src, "data", "chunk-000")
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".parquet"))
    if args.episodes:
        files = files[:args.episodes]

    raw, quantized, zero, excess = [], [], [], []
    total_steps = ik_failures = grip_clipped = fallbacks = unsafe = 0
    max_jump = 0.0
    max_residual = max_ori_residual = 0.0
    ms_per_step = []
    deltas_30hz, deltas_10hz = [], []
    grip_traces = []

    print(f"Validating {len(files)} episodes...\n")
    for fname in files:
        joints14 = read_column(os.path.join(src_dir, fname), "action")
        eef16 = read_column(os.path.join(args.dst, "data", "chunk-000", fname), "action")
        assert joints14.shape[1] == BIMANUAL_DOF and eef16.shape[1] == 16
        assert len(joints14) == len(eef16)

        # FK ignores joint ranges, so widening them for IK must not perturb the
        # training data this anchor has to agree with.
        anchor_err = np.abs(conv.latch_anchor(joints14[0]) - eef16[0]).max()
        assert anchor_err < 1e-6, f"anchor disagrees with dataset frame 0 by {anchor_err}"

        r = run_episode(joints14, eef16, conv)
        raw.append(r["arm_err"])
        excess.append(r["excess_step"])
        grip_traces.append(joints14[:, GRIPPERS])
        total_steps += r["stats"]["steps"]
        ik_failures += r["stats"]["ik_failures"]
        grip_clipped += r["stats"]["gripper_clipped"]
        fallbacks += r["stats"]["ik_fallbacks"]
        unsafe += r["stats"]["unsafe_jumps"]
        max_jump = max(max_jump, r["stats"]["max_fallback_jump_rad"])
        max_residual = max(max_residual, r["stats"]["max_pos_residual_m"])
        max_ori_residual = max(max_ori_residual, r["stats"]["max_ori_residual_rad"])
        ms_per_step.append(r["ms_per_step"])

        # Gripper must be bit-exact: it is passed through, never relativized.
        assert r["grip_err"].max() < 1e-6, f"gripper not passthrough in {fname}"

        cmd = r["joints_out"][:, ARM_JOINTS]
        deltas_30hz.append(np.abs(np.diff(cmd, axis=0)))
        deltas_10hz.append(np.abs(np.diff(cmd[::3], axis=0)))

        quantized.append(run_episode(joints14, eef16, conv, quant=(q01, q99))["arm_err"])
        zero.append(run_episode(joints14[:200], eef16[:200], conv, zero_seed=True)["arm_err"])

    print("1. Chain round-trip -- pose accuracy is the correctness metric")
    print(f"   IK failures          : {ik_failures} / {total_steps}")
    print(f"   max FK pos residual  : {max_residual * 1000:.4f} mm")
    print(f"   max FK ori residual  : {max_ori_residual:.2e} rad")
    print(f"   gripper clipped      : {grip_clipped}")
    print(f"   IK seed fallbacks    : {fallbacks}  (stalled seeds rescued by re-seeding)")
    print(f"   -> unsafe jumps      : {unsafe}  (branch switch > {conv.max_jump_rad} rad; "
          f"largest {max_jump:.4f} rad)")
    ex = np.concatenate(excess)
    print(f"   excess step vs demo  : p99={np.percentile(ex, 99):.4f}  max={ex.max():.4f} rad")

    print("\n2. Joint agreement with the demonstration (informational, not pass/fail)")
    for label, errs in (("exact", raw), ("through norm_stat quantize+clip", quantized)):
        st = summarize(label, np.concatenate(errs))
        print(f"   {st['name']:32s} p50={st['p50']:.2e}  p99={st['p99']:.2e}  max={st['max']:.2e}")
    print("   A tail here is expected and benign: a 6-DOF arm reaches one pose from")
    print("   several configurations. Where the demonstration pressed a joint against")
    print("   its own limit and held, IK drifts smoothly to an equivalent solution that")
    print("   hits the same pose without pinning the joint. Judge by pose + continuity.")

    print("\n3. Seeding and IK cost")
    s_seq = summarize("sequential seed", np.concatenate([e[:199] for e in raw]))
    s_zero = summarize("anchor-reseeded", np.concatenate(zero))
    print(f"   {s_seq['name']:20s} p99={s_seq['p99']:.2e}  max={s_seq['max']:.2e}")
    print(f"   {s_zero['name']:20s} p99={s_zero['p99']:.2e}  max={s_zero['max']:.2e}")
    mean_ms = float(np.mean(ms_per_step))
    print(f"   {mean_ms:.2f} ms/step (both arms) -> {mean_ms * 48:.0f} ms per 48-action "
          f"chunk, budget 1600 ms")

    print("\n4. Gripper polarity (expect ~1.0 = open at episode start, dips on grasp)")
    traces = np.concatenate(grip_traces)
    starts = np.stack([g[0] for g in grip_traces])
    print(f"   value range   : [{traces.min():.3f}, {traces.max():.3f}]")
    print(f"   at frame 0    : mean={starts.mean():.3f}  min={starts.min():.3f}")
    print(f"   verdict       : {'0=closed/1=open, matches YAM' if starts.mean() > 0.5 else 'INVERTED vs YAM -- fix in converter'}")

    print("\n5. Per-step arm-joint delta budget (for --max-joint-delta, rad)")
    for label, d in (("30 Hz (full)", deltas_30hz), ("10 Hz (every 3rd)", deltas_10hz)):
        flat = np.concatenate(d).reshape(-1)
        print(f"   {label:18s} p99={np.percentile(flat, 99):.4f}  "
              f"p99.99={np.percentile(flat, 99.99):.4f}  max={flat.max():.4f}")


if __name__ == "__main__":
    main()
