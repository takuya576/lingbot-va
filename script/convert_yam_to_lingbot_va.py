#!/usr/bin/env python3
"""Convert Datasets/yam-real-world-lerobot/cover_blocks (raw joint-space LeRobot v2.1
dataset) into a LingBot-VA-ready dataset that matches the robotwin EEF-pose convention:
16-dim observation.state/action (xyz+quat+gripper per arm), action_config segmentation
added to meta/episodes.jsonl, and a recomputed norm_stat (q01/q99) ready to paste into
a wan_va config.

Joint angles are converted to end-effector pose via forward kinematics using the YAM
arm's MuJoCo model from the i2rt project (vendored as a submodule in the user's
`raiden` repo). This step does NOT touch video files (copied as-is) or extract Wan2.2
VAE latents (see script/extract_va_latents.py for that, run separately in the
lingbot-va torch env).

Run in a lightweight env with: numpy, scipy, pandas, pyarrow, mujoco==3.3.5, mink>=0.0.13
(no torch needed). Requires the i2rt submodule checked out:
    cd <raiden_repo> && git submodule update --init third_party/i2rt

Usage:
    python script/convert_yam_to_lingbot_va.py \
        --src Datasets/yam-real-world-lerobot/cover_blocks \
        --dst Datasets/yam-real-world-lingbot-va/cover_blocks \
        --i2rt-root /data2/takuya/raiden/third_party/i2rt/i2rt
"""
import argparse
import json
import os
import shutil
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared with the deployment-side IK. Training FK and deployment IK must use the same
# MuJoCo model or the round trip picks up a constant offset that neither side's tests
# would catch on its own -- see yam_bridge/__init__.py.
from yam_bridge.action_convert import relativize_action16  # noqa: E402
from yam_bridge.kinematics import build_fk  # noqa: E402

# Canonical 30-dim channel layout used across wan_va/configs/va_robotwin_cfg.py and
# wan_va/dataset/lerobot_latent_dataset.py: [0:7]=left EEF, [7:14]=right EEF,
# [28]=left gripper, [29]=right gripper (matches va_robotwin_cfg.used_action_channel_ids).
LEFT_EEF_CANON = slice(0, 7)
RIGHT_EEF_CANON = slice(7, 14)
LEFT_GRIPPER_CANON = 28
RIGHT_GRIPPER_CANON = 29

RAW16_NAMES = [
    'left_x', 'left_y', 'left_z', 'left_q1', 'left_q2', 'left_q3', 'left_q4',
    'left_gripper', 'right_x', 'right_y', 'right_z', 'right_q1', 'right_q2',
    'right_q3', 'right_q4', 'right_gripper'
]


def joints14_to_eef16(arr14: np.ndarray, fk) -> np.ndarray:
    """(N,14) [left_j1..6,left_grip,right_j1..6,right_grip] -> (N,16) EEF layout."""
    n = arr14.shape[0]
    out = np.zeros((n, 16), dtype=np.float32)
    for i in range(n):
        row = arr14[i]
        left_q, left_grip = row[:6], row[6]
        right_q, right_grip = row[7:13], row[13]
        t_left = fk(left_q)
        t_right = fk(right_q)
        out[i, 0:3] = t_left[:3, 3]
        out[i, 3:7] = R.from_matrix(t_left[:3, :3]).as_quat()
        out[i, 7] = left_grip
        out[i, 8:11] = t_right[:3, 3]
        out[i, 11:15] = R.from_matrix(t_right[:3, :3]).as_quat()
        out[i, 15] = right_grip
    return out


def make_chunks(length: int, chunk_frames: int, min_chunk_frames: int) -> list:
    """Split [0, length) into consecutive (start, end) windows of ~chunk_frames each.

    chunk_frames <= 0 means "one segment per episode" (the default -- see the
    --chunk-frames help text for why that matters for anchor consistency).

    A trailing remainder shorter than min_chunk_frames is merged into the previous
    chunk rather than forming its own tiny segment.
    """
    if chunk_frames <= 0 or length <= chunk_frames:
        return [(0, length)]
    bounds = list(range(0, length, chunk_frames))
    if length - bounds[-1] < min_chunk_frames:
        bounds.pop()
    starts = bounds
    ends = bounds[1:] + [length]
    return list(zip(starts, ends))


def write_parquet(dst_path: str, df: pd.DataFrame, state16: np.ndarray, action16: np.ndarray):
    n = len(df)
    table = pa.Table.from_pydict({
        "observation.state": pa.FixedSizeListArray.from_arrays(
            pa.array(state16.reshape(-1), type=pa.float32()), 16),
        "action": pa.FixedSizeListArray.from_arrays(
            pa.array(action16.reshape(-1), type=pa.float32()), 16),
        "timestamp": pa.array(df["timestamp"].to_numpy(), type=pa.float32()),
        "frame_index": pa.array(df["frame_index"].to_numpy(), type=pa.int64()),
        "episode_index": pa.array(df["episode_index"].to_numpy(), type=pa.int64()),
        "index": pa.array(df["index"].to_numpy(), type=pa.int64()),
        "task_index": pa.array(df["task_index"].to_numpy(), type=pa.int64()),
    })
    assert table.num_rows == n
    pq.write_table(table, dst_path)


def col_stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Datasets/yam-real-world-lerobot/cover_blocks")
    ap.add_argument("--dst", default="Datasets/yam-real-world-lingbot-va/cover_blocks")
    ap.add_argument("--i2rt-root", default="/data2/takuya/raiden/third_party/i2rt/i2rt")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--chunk-frames", type=int, default=0,
                     help="0 (default) = one action_config segment per episode. "
                          "Every segment is then anchored at episode frame 0, which "
                          "is exactly the anchor the deployment client uses: it "
                          "captures init_eef_pose once per rollout and never "
                          "re-anchors (evaluation/robotwin/eval_polict_client_openpi.py "
                          "lines 576/615). Mid-episode segments would still be "
                          "self-consistent ('a rollout starting mid-task'), but "
                          "whole-episode segments keep training and deployment "
                          "anchored identically. Set >0 only if max_seq_len cannot "
                          "fit whole episodes: packed_tokens = (k+1)*(240+8*stride) "
                          "where k+1 = ((seg_frames-1)//stride)//4 + 1, and stride = "
                          "ori_fps/target_fps from script/extract_va_latents.py. At "
                          "5fps (stride 6) the longest cover_blocks episode is 26496 "
                          "tokens, hence max_seq_len=26624 in va_yam_lora_train_cfg.")
    ap.add_argument("--min-chunk-frames", type=int, default=100,
                     help="Only used when --chunk-frames > 0: a trailing remainder "
                          "shorter than this is merged into the previous chunk "
                          "instead of forming its own tiny segment. Worst-case "
                          "segment length is then chunk_frames+min_chunk_frames-1, "
                          "which must still fit max_seq_len.")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    if os.path.exists(dst) and args.overwrite:
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(dst, "meta"), exist_ok=True)

    fk = build_fk(args.i2rt_root)

    # 1. Videos: hardlink-copy (byte-identical, no re-encoding needed at this stage).
    dst_videos = os.path.join(dst, "videos")
    if not os.path.exists(dst_videos):
        shutil.copytree(os.path.join(src, "videos"), dst_videos, copy_function=os.link)
        print(f"[videos] hardlinked into {dst_videos}")

    # 2. Load source meta.
    with open(os.path.join(src, "meta", "episodes.jsonl")) as f:
        src_episodes = [json.loads(l) for l in f]
    with open(os.path.join(src, "meta", "info.json")) as f:
        src_info = json.load(f)
    src_stats_by_ep = {}
    with open(os.path.join(src, "meta", "episodes_stats.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            src_stats_by_ep[d["episode_index"]] = d

    new_episodes = []
    new_stats_lines = []
    all_rel_actions = []

    for ep in src_episodes:
        idx = ep["episode_index"]
        length = ep["length"]
        fname = f"episode_{idx:06d}.parquet"
        df = pd.read_parquet(os.path.join(src, "data", "chunk-000", fname))
        assert len(df) == length

        state14 = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action14 = np.stack(df["action"].to_numpy()).astype(np.float32)
        state16 = joints14_to_eef16(state14, fk)
        action16 = joints14_to_eef16(action14, fk)

        write_parquet(os.path.join(dst, "data", "chunk-000", fname), df, state16, action16)

        task_text = ep["tasks"][0]
        chunks = make_chunks(length, args.chunk_frames, args.min_chunk_frames)
        action_config = []
        for c_start, c_end in chunks:
            action_config.append({
                "start_frame": c_start,
                "end_frame": c_end,
                "action_text": task_text,
                "skill": "",
            })
            # norm_stat must be computed on the relative-pose-transformed action
            # PER SEGMENT (relative to that segment's own first frame), since that's
            # what _action_post_process applies before normalization.
            all_rel_actions.append(relativize_action16(action16[c_start:c_end]))

        new_episodes.append({
            "episode_index": idx,
            "tasks": ep["tasks"],
            "length": length,
            "action_config": action_config,
        })

        src_stats = src_stats_by_ep[idx]["stats"]
        new_stats = dict(src_stats)
        new_stats["observation.state"] = col_stats(state16)
        new_stats["action"] = col_stats(action16)
        new_stats_lines.append({"episode_index": idx, "stats": new_stats})

        print(f"[convert] episode {idx:06d} ({length} frames, {len(chunks)} chunks) done")

    # 3. meta/info.json -- update state/action feature blocks to 16-dim EEF layout.
    new_info = deepcopy(src_info)
    for key in ("observation.state", "action"):
        new_info["features"][key]["shape"] = [16]
        new_info["features"][key]["names"] = list(RAW16_NAMES)
    with open(os.path.join(dst, "meta", "info.json"), "w") as f:
        json.dump(new_info, f, indent=4)

    # 4. meta/tasks.jsonl -- copy unchanged.
    shutil.copy(os.path.join(src, "meta", "tasks.jsonl"), os.path.join(dst, "meta", "tasks.jsonl"))

    # 5. meta/episodes.jsonl (+ action_config) and meta/episodes_ori.jsonl (lineage).
    with open(os.path.join(dst, "meta", "episodes.jsonl"), "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")
    shutil.copy(os.path.join(src, "meta", "episodes.jsonl"), os.path.join(dst, "meta", "episodes_ori.jsonl"))

    # 6. meta/episodes_stats.jsonl
    with open(os.path.join(dst, "meta", "episodes_stats.jsonl"), "w") as f:
        for line in new_stats_lines:
            f.write(json.dumps(line) + "\n")

    # 7. norm_stat (q01/q99), computed on relative-pose-transformed actions, scattered
    #    into the canonical 30-dim layout. Grippers are NOT relativized (pass-through,
    #    already normalized [0,1] like robotwin's).
    all_rel = np.concatenate(all_rel_actions, axis=0)  # (total_frames, 16), raw order
    q01_16 = np.percentile(all_rel, 1, axis=0)
    q99_16 = np.percentile(all_rel, 99, axis=0)

    q01_30 = np.zeros(30)
    q99_30 = np.zeros(30)
    q01_30[LEFT_EEF_CANON] = q01_16[0:7]
    q99_30[LEFT_EEF_CANON] = q99_16[0:7]
    q01_30[RIGHT_EEF_CANON] = q01_16[8:15]
    q99_30[RIGHT_EEF_CANON] = q99_16[8:15]
    q01_30[LEFT_GRIPPER_CANON] = q01_16[7]
    q99_30[LEFT_GRIPPER_CANON] = q99_16[7]
    q01_30[RIGHT_GRIPPER_CANON] = q01_16[15]
    q99_30[RIGHT_GRIPPER_CANON] = q99_16[15]

    norm_stat = {"q01": q01_30.tolist(), "q99": q99_30.tolist()}
    with open(os.path.join(dst, "norm_stat.json"), "w") as f:
        json.dump(norm_stat, f, indent=2)

    print("\n=== norm_stat (paste into wan_va/configs/va_yam_lora_train_cfg.py) ===")
    print("va_yam_lora_train_cfg.norm_stat = {")
    print(f"    'q01': {q01_30.tolist()},")
    print(f"    'q99': {q99_30.tolist()},")
    print("}")
    print(f"\nDone. Converted dataset written to: {dst}")
    print("Next: copy empty_emb.pt from Datasets/robotwin-clean-and-aug-lerobot/, then "
          "run script/extract_va_latents.py to produce latents/.")


if __name__ == "__main__":
    main()
