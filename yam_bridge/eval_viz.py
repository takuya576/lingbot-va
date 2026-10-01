# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Shared rollout visualisation: trajectory plots and imagined-vs-real video.

Used by both evaluation paths, which is why this lives under `yam_bridge/` rather than
in one of the scripts -- `script/run_yam_policy.py` would otherwise have to import
`script/eval_yam_openloop.py`, dragging in that module's dataset-video decoding and its
argparse-shaped assumptions.

  - `script/eval_yam_openloop.py` plots recorded ground truth against the policy.
  - `script/run_yam_policy.py` plots the policy against what IK actually commanded and
    against what the arms measurably did.

The two differ only in how many series they hand to `plot_trajectories`, so the function
takes a list rather than a fixed pair.
"""
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
# Mosaic geometry copied from the server's _encode_obs (wan_va_server.py:326-376) with
# va_robotwin_cfg height/width = 256/320: cam_high at full size, each wrist at half,
# wrists side by side ABOVE cam_high. The result is 384x320 -- exactly the size of the
# canvas the VAE decodes, so the two panels line up pixel for pixel.
MOSAIC_H, MOSAIC_W = 256, 320
# 16-dim EEF layout: [l_xyz, l_quat, l_grip, r_xyz, r_quat, r_grip]
ARMS = {"left": (slice(0, 3), slice(3, 7), 7), "right": (slice(8, 11), slice(11, 15), 15)}


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


def _quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """(T,4) xyzw -> (T,3) roll/pitch/yaw in degrees, tolerating non-unit input.

    The model's raw output is not guaranteed unit-norm, and `R.from_quat` would either
    renormalise silently or raise on a zero row; normalising here keeps the plot honest
    about what was actually emitted without crashing on a degenerate step.
    """
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    return R.from_quat(quat / np.where(norm > 1e-12, norm, 1.0)).as_euler(
        "xyz", degrees=True)


def plot_trajectories(series: Sequence[dict], bounds: List[int], out_path: str,
                      title: str, subtitle: Optional[str] = None) -> None:
    """Overlay any number of 16-dim EEF trajectories, four panels, Euler for rotation.

    Layout follows `visualize_action_step` in
    evaluation/robotwin/eval_polict_client_openpi.py:88-135 (that module cannot be
    imported -- it chdir's into a hardcoded RoboTwin checkout at import time).

    The plot earns its place by showing *where* divergence starts: an aggregate error
    cannot distinguish "uniformly slightly off" from "tracks perfectly until the grasp".
    Every series is drawn at its true alignment; nothing is shifted to make them agree.

    `series` is a list of dicts:
      name  -- legend label
      data  -- (T, 16) array in the relative-EEF layout above
      t     -- optional (T,) x positions. Omit for a dense per-step series; supply it
               for a sparse one (e.g. measured state, sampled once per keyframe and
               filtered), which is then drawn as markers instead of a line.
      style -- optional matplotlib kwargs (ls, marker, lw, ms, alpha)

    Colour encodes the channel and linestyle encodes the series, so the channel legend
    lists each channel once -- listing every channel for every series multiplied the
    entries and covered the data. With more than two series the linestyle mapping stops
    being guessable, so a second legend spells it out.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Distinct without relying on colour, which is already spent on the channel.
    default_styles = [
        {"ls": "-", "lw": 1.6},
        {"ls": "--", "lw": 1.4},
        {"ls": ":", "lw": 1.6},
        {"ls": "-.", "lw": 1.3},
    ]
    resolved = []
    for i, s in enumerate(series):
        data = np.asarray(s["data"], dtype=np.float64)
        t = np.asarray(s["t"]) if s.get("t") is not None else np.arange(len(data))
        style = dict(default_styles[i % len(default_styles)])
        style.update(s.get("style") or {})
        if s.get("t") is not None and "marker" not in style:
            # Sparse by construction: joining the dots would draw straight lines across
            # the samples that were dropped, inventing motion that was never observed.
            style.update({"ls": "none", "marker": "o", "ms": 3.0, "alpha": 0.85})
        resolved.append({"name": s["name"], "data": data, "t": t, "style": style})

    # constrained_layout reserves the room the outside legends need.
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=110, sharex=True,
                             layout="constrained")
    pos_lbls, pos_cols = ("x", "y", "z", "gripper"), ("tab:red", "tab:green", "tab:blue", "k")
    rot_lbls, rot_cols = ("roll", "pitch", "yaw"), ("tab:orange", "tab:purple", "tab:cyan")

    for col, (arm, (xyz, quat, grip)) in enumerate(ARMS.items()):
        ax_p, ax_r = axes[0, col], axes[1, col]

        for s in resolved:
            d, t, st = s["data"], s["t"], s["style"]
            if not len(d):
                continue
            for i, c in enumerate(pos_cols[:3]):
                ax_p.plot(t, d[:, xyz][:, i], color=c, **st)
            ax_p.plot(t, d[:, grip], color="k", **st)
            rpy = _quat_to_rpy(d[:, quat])
            for i, c in enumerate(rot_cols):
                ax_r.plot(t, rpy[:, i], color=c, **st)

        ax_p.set_title(f"{arm} arm — relative translation (m) + gripper")
        ax_p.set_ylabel("m / normalised")
        ax_r.set_title(f"{arm} arm — relative rotation (deg)")
        ax_r.set_xlabel("control step (30 Hz)")
        ax_r.set_ylabel("deg")

        for ax, lbls, cols in ((ax_p, pos_lbls, pos_cols), (ax_r, rot_lbls, rot_cols)):
            # Error that jumps at a boundary rather than drifting within a chunk points
            # at the world model losing track between chunks, not at the action head.
            for b in bounds[1:]:
                ax.axvline(b, color="0.6", lw=0.9, ls=":", zorder=0)
            ax.legend(handles=[Line2D([0], [0], color=c, lw=2, label=l)
                               for l, c in zip(lbls, cols)],
                      loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      fontsize=8, frameon=False)
            ax.grid(alpha=0.25)

    # Series legend, once for the figure: with three series the linestyle mapping can no
    # longer be carried by a sentence in the suptitle.
    axes[0, 1].legend(
        handles=[Line2D([0], [0], color="0.25", label=s["name"],
                        **{k: v for k, v in s["style"].items() if k != "alpha"})
                 for s in resolved],
        loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize=8,
        frameon=False, title="series", title_fontsize=8)

    head = f"{title}   —   dotted verticals = chunk boundaries"
    fig.suptitle(head if subtitle is None else f"{head}\n{subtitle}", fontsize=11)
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


def save_imagination(chunks: List[tuple], out_path: str, labels: bool = True,
                     real_caption: str = "ground truth",
                     first_frame_is_conditioning: bool = True) -> None:
    """Imagined world-model canvas beside the real frame from the same moment.

    The cheapest high-signal check that the observation pipeline is right: a wrong camera
    order, wrong aspect ratio or mis-tiled T-shape mosaic is obvious on sight, where the
    action metrics would merely read "somewhat off" and get blamed on the adapter.

    `chunks` is [(imagined (T,H,W,3) float in [0,1], [real keyframe dicts])]. The two streams
    have different lengths -- the VAE decodes a chunk to a handful of frames while the
    chunk spans 4 or 8 keyframes -- so imagined frame j is matched to the proportionally
    nearest real keyframe rather than to a single static one.

    Frame counts per chunk depend on whether the server decodes continuously. With the
    streaming decoder (`WanVAEStreamingDecoder`) the first latent of the ROLLOUT expands
    to 1 pixel frame and every latent after it to 4, so at frame_chunk_size=2 a rollout
    runs 5, 8, 8, ... With the older cold-per-chunk decode every chunk restarted the
    decoder and yielded 5. Nothing here depends on which: the proportional match adapts.

    `labels` draws which panel is which and the chunk/frame counters. On by default --
    without them the two panels are easy to confuse and there is no way to tell which
    chunk you are looking at. Turn it off for a clean render, e.g. for a figure.

    `real_caption` names the right-hand panel. Open-loop replays recorded episodes, so
    there it really is ground truth; closed-loop is watching the robot's own live camera
    while the policy drives it, which is an observation and not a reference.

    `first_frame_is_conditioning` captions the very first frame differently. Latent frame
    0 of the first chunk is the conditioning frame the server rebuilds from the initial
    observation (`wan_va_server.py:492-493`), so its decode is a VAE reconstruction of a
    real frame, not something the world model imagined -- and it is the one frame in the
    video where a discrepancy means the encode/decode round trip is wrong rather than the
    policy. Pass False when `chunks` does not start at the beginning of a rollout.
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
            is_cond = first_frame_is_conditioning and ci == 1 and j == 0
            if labels:
                # ASCII only, and short: cv2.putText's Hershey fonts render anything
                # else as "?", and it clips silently at the panel edge rather than
                # wrapping -- at scale 0.5 the 320px panel holds ~35 characters.
                left = _label(f, "conditioning (VAE recon)" if is_cond else "imagined")
                # Which chunk is on screen. Divergence usually grows within a chunk and
                # is partly reset when the next observation lands, so the chunk index is
                # what makes the video readable against the trajectory plot's boundaries.
                # Kept at the top: the bottom strip of a frame is where player scrubbers
                # and control overlays sit, which hides anything drawn there.
                # "video frame", not keyframe: the two differ and conflating them is
                # confusing. A chunk is 48 actions = 2 latent frames, fed by 8 recorded
                # keyframes (the encoder takes 4 pixel frames per latent). Decoding those
                # 2 latents yields 8 pixel frames once the decoder cache is warm, and 5
                # for the very first chunk -- the causal VAE emits a single frame for the
                # first latent of a stream and 4 for every one after it.
                # Both counters are 1-based so they read the same way.
                _label(left, f"chunk {ci}/{total}   video frame {j + 1}/{n}",
                       org=(6, 38), copy=False)
                right = _label(gt, real_caption)
            divider = np.full((f.shape[0], 2, 3), 255, np.uint8)
            frames.append(np.concatenate([left, divider, right], axis=1))
    _write_mp4(frames, out_path)
