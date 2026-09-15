#!/usr/bin/env python3
"""Open-loop, teacher-forced evaluation of the YAM policy against recorded episodes.

Drives the real policy-server protocol with **recorded** observations and compares the
predicted trajectory against ground truth. No robot involved, so this is the zero-risk
half of pre-deployment verification, and the only test that says anything about whether
the model learned: replaying a converted dataset on hardware exercises the robot path but
cannot tell you the policy is broken.

It also measures IK behaviour on **model-distribution** poses. `validate_yam_action_convert.py`
could only use demonstration poses, and the model's outputs sit somewhere else; the
`unsafe_jumps` and delta numbers from here are the ones that should set
`--max-joint-delta`.

Teacher forcing means the world model is advanced with recorded keyframes rather than
consequences of its own predictions, so this measures per-chunk prediction quality under
the training distribution. It does **not** measure closed-loop drift.

Caveat: all 74 episodes were trained on -- there is no held-out set. A good result rules
out gross failure (unlearned adapter, wrong norm stats, broken observation pipeline). It
is not a success-rate estimate.

Usage (needs the policy server up: NGPU=1 CONFIG_NAME=yam bash script/run_launch_va_server_sync.sh)::

    /data2/takuya/raiden/.venv/bin/python script/eval_yam_openloop.py --episodes 2
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yam_bridge.action_convert import (  # noqa: E402
    DEFAULT_I2RT_ROOT, YAM_DOF, YamActionConverter, relativize_action16,
)
from yam_bridge.kinematics import ARM_DOF  # noqa: E402
from yam_bridge.va_client import VAClient  # noqa: E402

CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
ARM_JOINTS = np.r_[0:ARM_DOF, YAM_DOF:YAM_DOF + ARM_DOF]
# Mosaic geometry copied from the server's _encode_obs (wan_va_server.py:326-376) with
# va_robotwin_cfg height/width = 256/320: cam_high at full size, each wrist at half,
# wrists side by side ABOVE cam_high. The result is 384x320 -- exactly the size of the
# canvas the VAE decodes, so the two panels line up pixel for pixel.
MOSAIC_H, MOSAIC_W = 256, 320
# 16-dim EEF layout: [l_xyz, l_quat, l_grip, r_xyz, r_quat, r_grip]
ARMS = {"left": (slice(0, 3), slice(3, 7), 7), "right": (slice(8, 11), slice(11, 15), 15)}


def read_column(path: str, column: str) -> np.ndarray:
    table = pq.read_table(path, columns=[column])
    return np.stack(table.column(column).to_numpy(zero_copy_only=False)).astype(np.float64)


def decode_frames(video_dir: str, ep: int, wanted: set) -> Dict[int, np.ndarray]:
    """Decode only the frames in `wanted` from one episode video, as RGB uint8.

    cv2 decodes BGR while the dataset was written from RGB, so the channel flip is
    required, not cosmetic -- feeding BGR would swap red and blue for the whole rollout
    and look like a mediocre policy rather than a bug.
    """
    path = os.path.join(video_dir, f"episode_{ep:06d}.mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    out, idx, last = {}, 0, max(wanted)
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            out[idx] = np.ascontiguousarray(frame[..., ::-1])
        idx += 1
    cap.release()
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"{path}: frames {sorted(missing)[:5]}... not decodable")
    return out


def tshape_mosaic(frame: Dict[str, np.ndarray]) -> np.ndarray:
    """Tile the three recorded views the way the server composes its latent canvas.

    Rendering ground truth in the same T-shape as the imagination is what makes the
    comparison readable: both wrist views and the scene view sit in the same places at
    the same scale, so a mis-tiled or mis-ordered mosaic shows up as a mismatch between
    the panels rather than having to be judged from one panel alone.
    """
    high = cv2.resize(frame[CAM_KEYS[0]], (MOSAIC_W, MOSAIC_H))
    half_w, half_h = MOSAIC_W // 2, MOSAIC_H // 2
    left = cv2.resize(frame[CAM_KEYS[1]], (half_w, half_h))
    right = cv2.resize(frame[CAM_KEYS[2]], (half_w, half_h))
    return np.concatenate([np.concatenate([left, right], axis=1), high], axis=0)


def _label(img: np.ndarray, text: str, org=(6, 18), copy: bool = True) -> np.ndarray:
    """Draw text with a dark outline so it stays legible over any background."""
    out = img.copy() if copy else img
    for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thick, cv2.LINE_AA)
    return out


def rotation_error_deg(pred_quat: np.ndarray, gt_quat: np.ndarray) -> np.ndarray:
    """Geodesic angle between two xyzw quaternion sequences, in degrees."""
    pred = pred_quat / np.linalg.norm(pred_quat, axis=1, keepdims=True)
    gt = gt_quat / np.linalg.norm(gt_quat, axis=1, keepdims=True)
    return np.degrees((R.from_quat(gt).inv() * R.from_quat(pred)).magnitude())


def best_lag(pred: np.ndarray, gt_all: np.ndarray, base: int, span: int = 8) -> dict:
    """Scan frame offsets and report the whole error curve, not just the argmin.

    The harness alignment is pinned by an oracle test (a perfect prediction yields
    exactly zero error at lag 0), so a nonzero argmin here is *not* an indexing bug --
    it means the scan has no sharp minimum because the prediction is not tracking, and
    the argmin is then close to noise. Reporting `err@0`, the best, and whether the best
    sits at the edge of the scan is what makes the two cases distinguishable.
    """
    errs = {}
    for lag in range(-span, span + 1):
        idx = base + np.arange(len(pred)) + lag
        if idx[0] < 0 or idx[-1] >= len(gt_all):
            continue
        errs[lag] = float(np.linalg.norm(
            pred[:, 0:3] - gt_all[idx][:, 0:3], axis=1).mean() * 1000)
    lag = min(errs, key=errs.get)
    return {"lag": lag, "err_mm": errs[lag], "err0_mm": errs.get(0, float("nan")),
            "at_edge": abs(lag) == span, "curve": errs}


def plot_trajectories(pred: np.ndarray, gt: np.ndarray, bounds: List[int],
                      out_path: str, title: str) -> None:
    """Ground truth vs prediction, four panels, quaternions shown as Euler angles.

    Layout follows `visualize_action_step` in
    evaluation/robotwin/eval_polict_client_openpi.py:88-135 (that module cannot be
    imported -- it chdir's into a hardcoded RoboTwin checkout at import time).

    The plot earns its place by showing *where* divergence starts: an aggregate error
    cannot distinguish "uniformly slightly off" from "tracks perfectly until the grasp".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=110, sharex=True)
    t = np.arange(len(pred))
    for col, (arm, (xyz, quat, grip)) in enumerate(ARMS.items()):
        ax_p, ax_r = axes[0, col], axes[1, col]
        for i, (lbl, c) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
            ax_p.plot(t, gt[:, xyz][:, i], color=c, lw=1.6, label=f"{lbl} gt")
            ax_p.plot(t, pred[:, xyz][:, i], color=c, lw=1.4, ls="--", label=f"{lbl} pred")
        ax_p.plot(t, gt[:, grip], color="k", lw=1.6, label="grip gt")
        ax_p.plot(t, pred[:, grip], color="k", lw=1.4, ls="--", label="grip pred")
        ax_p.set_title(f"{arm} arm — relative translation (m) + gripper")
        ax_p.set_ylabel("m / normalised")

        gt_rpy = R.from_quat(gt[:, quat]).as_euler("xyz", degrees=True)
        pr = pred[:, quat] / np.linalg.norm(pred[:, quat], axis=1, keepdims=True)
        pred_rpy = R.from_quat(pr).as_euler("xyz", degrees=True)
        for i, (lbl, c) in enumerate(zip(("roll", "pitch", "yaw"),
                                         ("tab:orange", "tab:purple", "tab:cyan"))):
            ax_r.plot(t, gt_rpy[:, i], color=c, lw=1.6, label=f"{lbl} gt")
            ax_r.plot(t, pred_rpy[:, i], color=c, lw=1.4, ls="--", label=f"{lbl} pred")
        ax_r.set_title(f"{arm} arm — relative rotation (deg)")
        ax_r.set_xlabel("control step (30 Hz)")
        ax_r.set_ylabel("deg")

        for ax in (ax_p, ax_r):
            # Error that jumps at a boundary rather than drifting within a chunk points
            # at the world model losing track between chunks, not at the action head.
            for b in bounds[1:]:
                ax.axvline(b, color="0.6", lw=0.9, ls=":", zorder=0)
            ax.legend(fontsize=7, ncol=4, loc="upper left")
            ax.grid(alpha=0.25)

    fig.suptitle(f"{title}   (solid = ground truth, dashed = policy; "
                 f"grey lines = chunk boundaries)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_mp4(frames: List[np.ndarray], out_path: str, fps: float = 5.0) -> None:
    """Encode RGB frames to H.264, falling back to cv2's mp4v if ffmpeg is absent.

    cv2 cannot write `avc1` -- the FFmpeg bundled in the opencv wheels carries no H.264
    encoder -- and its `mp4v` output (MPEG-4 Part 2) is a valid file that Chromium-based
    viewers such as Cursor and VS Code cannot decode, so it silently fails to preview.
    Piping to the ffmpeg binary with libx264 + yuv420p + faststart is what makes the
    result playable in an editor and a browser.
    """
    if not frames:
        return
    h, w = frames[0].shape[:2]
    # yuv420p needs even dimensions.
    if h % 2 or w % 2:
        h, w = h - (h % 2), w - (w % 2)
        frames = [f[:h, :w] for f in frames]

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
            input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode == 0:
            return
        print(f"  ffmpeg failed ({proc.stderr.decode()[:200]}); falling back to mp4v")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"  could not open {out_path} for writing; skipping video")
        return
    for f in frames:
        writer.write(np.ascontiguousarray(f[..., ::-1]))   # cv2 wants BGR
    writer.release()
    print("  NOTE: wrote mp4v (no ffmpeg found) -- may not preview in Cursor/VS Code")


def save_imagination(chunks: List[tuple], out_path: str,
                     labels: bool = True) -> None:
    """Imagined world-model canvas beside the recorded frame from the same moment.

    The cheapest high-signal check that the observation pipeline is right: a wrong camera
    order, wrong aspect ratio or mis-tiled T-shape mosaic is obvious on sight, where the
    action metrics would merely read "somewhat off" and get blamed on the adapter.

    `chunks` is [(imagined (T,H,W,3) float in [0,1], [real keyframe dicts])]. The two streams
    have different lengths -- the VAE decodes a chunk to a handful of frames while the
    chunk spans 4 or 8 keyframes -- so imagined frame j is matched to the proportionally
    nearest real keyframe rather than to a single static one.

    `labels` draws which panel is which and the chunk/frame counters. On by default --
    without them the two panels are easy to confuse and there is no way to tell which
    chunk you are looking at. Turn it off for a clean render, e.g. for a figure.
    """
    frames = []
    total = len(chunks)
    for ci, (imagined, reals) in enumerate(chunks, start=1):
        if imagined is None or not len(reals):
            continue
        canvas = (np.clip(imagined, 0, 1) * 255).astype(np.uint8)
        n = len(canvas)
        for j, f in enumerate(canvas):
            k = min(round(j * (len(reals) - 1) / max(1, n - 1)), len(reals) - 1)
            gt = tshape_mosaic(reals[k])
            if gt.shape[:2] != f.shape[:2]:      # decoded canvas should already match
                gt = cv2.resize(gt, (f.shape[1], f.shape[0]))
            left, right = f, gt
            if labels:
                left = _label(f, "imagined")
                # Which chunk is on screen. Divergence usually grows within a chunk and
                # is partly reset when the next observation lands, so the chunk index is
                # what makes the video readable against the trajectory plot's boundaries.
                # Kept at the top: the bottom strip of a frame is where player scrubbers
                # and control overlays sit, which hides anything drawn there.
                # Both counters are 1-based so they read the same way.
                _label(left, f"chunk {ci}/{total}   frame {j + 1}/{n}",
                       org=(6, 38), copy=False)
                right = _label(gt, "ground truth")
            divider = np.full((f.shape[0], 2, 3), 255, np.uint8)
            frames.append(np.concatenate([left, divider, right], axis=1))
    _write_mp4(frames, out_path)


def evaluate(ep: int, args, va: VAClient, conv: YamActionConverter) -> dict:
    src = os.path.join(args.src, "data", "chunk-000", f"episode_{ep:06d}.parquet")
    dst = os.path.join(args.dst, "data", "chunk-000", f"episode_{ep:06d}.parquet")
    joints14 = read_column(src, "action")
    gt_rel = relativize_action16(read_column(dst, "action"))

    with open(os.path.join(args.dst, "meta", "episodes.jsonl")) as f:
        meta = {json.loads(l)["episode_index"]: json.loads(l) for l in f}
    prompt = meta[ep]["tasks"][0]

    # Chunk 0 executes only latent frame 1 (24 steps); later chunks execute 48. A
    # keyframe follows every 6 steps, which is 5 Hz -- the rate the latents were
    # extracted at.
    n_chunks = args.max_chunks
    total = 24 + 48 * (n_chunks - 1)
    if total + 8 >= len(joints14):
        n_chunks = max(1, (len(joints14) - 32) // 48)
        total = 24 + 48 * (n_chunks - 1)
    keyframe_steps = list(range(6, total + 1, 6))
    frames = decode_frames_all(args.dst, ep, set([0] + keyframe_steps))

    va.reset(prompt)
    conv.reset_stats()
    conv.latch_anchor(joints14[0])

    first_obs = frames[0]
    preds, cmds_all, bounds, videos = [], [], [], []
    step = 0
    t_total = 0.0

    for c in range(n_chunks):
        t0 = time.perf_counter()
        got = va.infer(first_obs, want_video=args.video)
        action, imagined = got if args.video else (got, None)
        t_total += time.perf_counter() - t0

        start_idx = 1 if c == 0 else 0
        executed = action[:, start_idx:, :]
        bounds.append(step)   # cumulative control steps before this chunk
        preds.append(executed.transpose(1, 2, 0).reshape(-1, 16))
        cmds_all.append(conv.convert_chunk(executed))

        n_exec = preds[-1].shape[0]
        ks = [step + s for s in range(6, n_exec + 1, 6)]
        step += n_exec
        # frames is {frame_index: {cam_key: array}}, which is already the per-timestep
        # dict shape the server expects.
        va.advance([frames[idx] for idx in ks], action)

        if args.video:
            videos.append((imagined, [frames[k] for k in ks]))

    pred = np.concatenate(preds)
    cmds = np.concatenate(cmds_all)
    # Executed action j corresponds to ground-truth frame j. Training pads
    # frame_stride*4 = 24 zeros at the *front* before reshaping into latent frames
    # (lerobot_latent_dataset.py:268), so latent frame 0 holds the pad -- which is why
    # it is skipped -- and latent frame i covers raw frames (i-1)*24 .. i*24-1. Chunk 0's
    # single executed frame is therefore raw frames 0..23, not 1..24.
    scan = best_lag(pred, gt_rel, base=0)
    lag = scan["lag"]
    gt = gt_rel[lag: lag + len(pred)] if lag >= 0 else gt_rel[:len(pred)]
    n = min(len(gt), len(pred))
    pred, gt = pred[:n], gt[:n]

    res = {"episode": ep, "chunks": n_chunks, "steps": n, "lag": lag,
           "lag_scan": {k: scan[k] for k in ("err_mm", "err0_mm", "at_edge")},
           "infer_ms": t_total / n_chunks * 1e3, "stats": dict(conv.stats), "arms": {}}
    for arm, (xyz, quat, grip) in ARMS.items():
        terr = np.linalg.norm(pred[:, xyz] - gt[:, xyz], axis=1) * 1000
        rerr = rotation_error_deg(pred[:, quat], gt[:, quat])
        res["arms"][arm] = {
            "trans_mm": np.percentile(terr, [50, 99]).tolist() + [terr.max()],
            "rot_deg": np.percentile(rerr, [50, 99]).tolist() + [rerr.max()],
            "grip_mae": float(np.abs(pred[:, grip] - gt[:, grip]).mean()),
        }
    res["max_step_delta"] = float(np.abs(np.diff(cmds[:, ARM_JOINTS], axis=0)).max())

    os.makedirs(args.out, exist_ok=True)
    tag = f"episode_{ep:06d}"
    plot_trajectories(pred, gt, bounds, os.path.join(args.out, f"{tag}_traj.png"),
                      f"{tag}  (lag={lag})")
    if args.video and videos:
        save_imagination(videos, os.path.join(args.out, f"{tag}_imagined.mp4"),
                         labels=args.labels)
    return res


def decode_frames_all(dataset_root: str, ep: int, wanted: set) -> Dict[int, Dict[str, np.ndarray]]:
    """{frame_index: {cam_key: RGB array}} for the three cameras."""
    per_cam = {k: decode_frames(os.path.join(dataset_root, "videos", "chunk-000", k),
                                ep, wanted) for k in CAM_KEYS}
    return {i: {k: per_cam[k][i] for k in CAM_KEYS} for i in wanted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Datasets/yam-real-world-lerobot/cover_blocks")
    ap.add_argument("--dst", default="Datasets/yam-real-world-lingbot-va/cover_blocks")
    ap.add_argument("--i2rt-root", default=DEFAULT_I2RT_ROOT)
    ap.add_argument("--va-host", default="0.0.0.0")
    ap.add_argument("--va-port", type=int, default=29536)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-chunks", type=int, default=5)
    ap.add_argument("--out", default="eval_out/yam_openloop")
    ap.add_argument("--no-video", dest="video", action="store_false",
                    help="Skip the imagined-video dump (saves a server-side VAE decode).")
    ap.add_argument("--no-labels", dest="labels", action="store_false",
                    help="Render the video without the 'imagined'/'ground truth' captions "
                         "and the chunk/frame counters (on by default).")
    args = ap.parse_args()

    print(f"connecting to policy server {args.va_host}:{args.va_port} "
          "(blocks until it accepts)...")
    va = VAClient(host=args.va_host, port=args.va_port)
    conv = YamActionConverter(args.i2rt_root)

    results = []
    for ep in range(args.episodes):
        print(f"\n=== episode {ep} ===")
        r = evaluate(ep, args, va, conv)
        results.append(r)
        s = r["stats"]
        print(f"  {r['steps']} steps over {r['chunks']} chunks, "
              f"{r['infer_ms']:.0f} ms/chunk inference")
        sc = r["lag_scan"]
        if r["lag"] == 0:
            note = "(aligned)"
        elif sc["at_edge"]:
            note = ("<-- scan hit its edge: the prediction is not tracking, so this "
                    "offset is noise, not a shift")
        else:
            note = f"(err@0 {sc['err0_mm']:.1f} mm vs {sc['err_mm']:.1f} mm at best)"
        print(f"  lag scan: best offset {r['lag']:+d}  {note}")
        for arm, m in r["arms"].items():
            print(f"  {arm:5s} trans p50/p99/max {m['trans_mm'][0]:7.1f}/"
                  f"{m['trans_mm'][1]:7.1f}/{m['trans_mm'][2]:7.1f} mm   "
                  f"rot {m['rot_deg'][0]:5.1f}/{m['rot_deg'][1]:5.1f}/{m['rot_deg'][2]:5.1f} deg   "
                  f"grip MAE {m['grip_mae']:.3f}")
        print(f"  IK on model outputs: {s['ik_failures']} failures, "
              f"{s['ik_fallbacks']} fallbacks, {s['unsafe_jumps']} unsafe jumps, "
              f"max residual {s['max_pos_residual_m'] * 1000:.2f} mm")
        print(f"  max per-step joint delta: {r['max_step_delta']:.4f} rad "
              f"(sets --max-joint-delta)")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote plots, video and results.json to {args.out}/")


if __name__ == "__main__":
    main()
