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

Run in a lightweight env with: numpy, scipy, pyarrow, mujoco==3.3.5, mink>=0.0.13
(no torch needed). Requires the i2rt submodule checked out:
    cd <raiden_repo> && git submodule update --init third_party/i2rt

Usage:
    python script/convert_yam_to_lingbot_va.py \
        --src Datasets/yam-real-world-lerobot/cover_blocks \
        --dst Datasets/yam-real-world-lingbot-va/cover_blocks \
        --i2rt-root /data2/takuya/raiden/third_party/i2rt/i2rt

`--action-space joint` instead keeps the raw 14-dim joints (no FK, so no mujoco or
i2rt needed) and fits norm_stat on ABSOLUTE joint angles, scattered into the base
model's joint slots -- the configs with `action_space = 'joint'` read it. Latents
depend only on the videos and the action_config segmentation, never on actions, so
`--latents-from` copies them from an already-extracted EEF conversion of the same
--src instead of re-running extract_va_latents.py::

    python script/convert_yam_to_lingbot_va.py --action-space joint \
        --src Datasets/yam-real-world-lerobot/place_cube \
        --dst Datasets/yam-real-world-lingbot-va-joint/place_cube \
        --latents-from Datasets/yam-real-world-lingbot-va/place_cube \
        --norm-stat-episodes 0-49

Keep joint datasets OUT of Datasets/yam-real-world-lingbot-va/: a dataset_path pointed
at that parent makes construct_lerobot_multi_processor sweep in every dataset beneath
it.
"""
import argparse
import json
import os
import shutil
import sys
from copy import deepcopy

import numpy as np
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

# --action-space joint: the base model's joint slots, 7 per arm. YAM has 6 arm joints,
# so channels 20 and 27 stay unused (zero, masked out), as the base model's
# pretraining padded every <7-DoF arm. Order matches the raw 14-dim layout's arms.
LEFT_JOINT_CANON = slice(14, 20)
RIGHT_JOINT_CANON = slice(21, 27)

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


def read_list_column(table: pa.Table, column: str) -> np.ndarray:
    """Read one FixedSizeList column of a LeRobot parquet as (N, D) float32.

    Same access pattern as script/validate_yam_action_convert.py's `read_column`.
    """
    return np.stack(
        table.column(column).to_numpy(zero_copy_only=False)).astype(np.float32)


def write_parquet(dst_path: str, src_table: pa.Table, state: np.ndarray,
                   action: np.ndarray):
    """Rewrite one episode with the given (N, D) state/action, scalar columns carried over."""
    dim = action.shape[1]
    table = pa.Table.from_pydict({
        "observation.state": pa.FixedSizeListArray.from_arrays(
            pa.array(state.reshape(-1), type=pa.float32()), dim),
        "action": pa.FixedSizeListArray.from_arrays(
            pa.array(action.reshape(-1), type=pa.float32()), dim),
        "timestamp": src_table.column("timestamp").cast(pa.float32()),
        "frame_index": src_table.column("frame_index").cast(pa.int64()),
        "episode_index": src_table.column("episode_index").cast(pa.int64()),
        "index": src_table.column("index").cast(pa.int64()),
        "task_index": src_table.column("task_index").cast(pa.int64()),
    })
    assert table.num_rows == src_table.num_rows
    pq.write_table(table, dst_path)


def parse_episode_spec(spec: str) -> set:
    """'0-49' / '0-9,20,30-32' -> a set of episode indices. None -> None (all)."""
    if spec is None:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))  # inclusive, as written
        else:
            out.add(int(part))
    assert out, f"--norm-stat-episodes {spec!r} selected no episodes"
    return out


def video_cameras(info: dict) -> list:
    """The camera keys meta/info.json declares -- every segment needs a latent for each."""
    return sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")


def latent_problems(latents_root: str, new_episodes: list, cams: list,
                    source_root: str = None) -> list:
    """Every expected latent file under `latents_root` that is missing -- or, with
    `source_root`, that is the same file (inode) as its counterpart there.

    Checked against `cams` from the metadata, never against the camera directories
    actually present: a tree missing a whole camera directory would otherwise check out
    as complete, and _check_meta would then drop every segment that needs that camera.
    """
    problems = []
    for ep in new_episodes:
        for c in ep["action_config"]:
            name = f"episode_{ep['episode_index']:06d}_{c['start_frame']}_{c['end_frame']}.pth"
            for cam in cams:
                rel = os.path.join("chunk-000", cam, name)
                path = os.path.join(latents_root, rel)
                if not os.path.exists(path):
                    problems.append(f"missing {rel}")
                elif source_root is not None:
                    src = os.path.join(source_root, rel)
                    if os.path.exists(src) and os.path.samefile(path, src):
                        problems.append(f"hardlinked to the source: {rel}")
    return problems


def copy_latents(latents_from: str, dst: str, new_episodes: list, cams: list) -> None:
    """Copy `latents/` and `empty_emb.pt` from another conversion of the same --src.

    Only valid when that conversion segmented every episode identically: the latent
    files are keyed by (episode, start_frame, end_frame) and carry the segment's T5
    embedding of action_text. So the action_config is compared first, and then every
    segment's file is checked for -- _check_meta drops a segment whose latent is
    missing without saying so, which would shrink the training set silently.

    Copied, never hardlinked: extract_va_latents.py --overwrite rewrites latent files in
    place with torch.save, so a shared inode would let re-extracting either dataset
    silently rewrite -- or, interrupted, truncate -- the other's. The copy is cheap
    (714 MB for place_cube). It is built under latents.partial and renamed into place
    only once complete, so an interrupted run never leaves a latents/ that a rerun would
    take as finished; an existing latents/ is re-verified, and refused if incomplete or
    still sharing files with the source (an earlier version of this script hardlinked).
    """
    with open(os.path.join(latents_from, "meta", "episodes.jsonl")) as f:
        theirs = {e["episode_index"]: e["action_config"] for e in map(json.loads, f)}
    seg_key = lambda c: (c["start_frame"], c["end_frame"], c["action_text"])  # noqa: E731
    for ep in new_episodes:
        mine = [seg_key(c) for c in ep["action_config"]]
        other = [seg_key(c) for c in theirs.get(ep["episode_index"], [])]
        assert mine == other, (
            f"episode {ep['episode_index']}: action_config differs from "
            f"{latents_from} ({mine} vs {other}); its latents do not apply. Match "
            f"--chunk-frames/--min-chunk-frames, or extract latents for this dataset.")

    src_latents = os.path.join(latents_from, "latents")
    problems = latent_problems(src_latents, new_episodes, cams)
    assert not problems, (f"{src_latents} is itself incomplete ({len(problems)} "
                          f"problems, e.g. {problems[:3]}); extract its latents first.")

    dst_latents = os.path.join(dst, "latents")
    if not os.path.exists(dst_latents):
        partial = dst_latents + ".partial"
        if os.path.exists(partial):  # left by an interrupted run
            shutil.rmtree(partial)
        shutil.copytree(src_latents, partial)
        problems = latent_problems(partial, new_episodes, cams)
        assert not problems, f"copy came out incomplete: {problems[:3]}"
        os.rename(partial, dst_latents)
    problems = latent_problems(dst_latents, new_episodes, cams, source_root=src_latents)
    assert not problems, (
        f"{dst_latents} cannot be used ({len(problems)} problems, e.g. {problems[:3]}). "
        f"Delete it and rerun.")

    # Replaced atomically rather than skipped when present, so a hardlink left by an
    # earlier version is broken without touching the source's file.
    emb_tmp = os.path.join(dst, "empty_emb.pt.partial")
    shutil.copy2(os.path.join(latents_from, "empty_emb.pt"), emb_tmp)
    os.replace(emb_tmp, os.path.join(dst, "empty_emb.pt"))

    print(f"[latents] copied from {latents_from}: {len(cams)} cameras "
          f"({', '.join(cams)}), all "
          f"{sum(len(e['action_config']) for e in new_episodes)} segments present")


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
                          "tokens, which needs max_seq_len >= 26624.")
    ap.add_argument("--min-chunk-frames", type=int, default=100,
                     help="Only used when --chunk-frames > 0: a trailing remainder "
                          "shorter than this is merged into the previous chunk "
                          "instead of forming its own tiny segment. Worst-case "
                          "segment length is then chunk_frames+min_chunk_frames-1, "
                          "which must still fit max_seq_len.")
    ap.add_argument("--norm-stat-episodes", default=None, metavar="SPEC",
                     help="Restrict which episodes the q01/q99 fit is computed over, "
                          "e.g. '0-49' or '0-9,20,30-32'. Every episode is still "
                          "converted and written; only step 7's quantile fit is "
                          "narrowed. Set this to the training split whenever the "
                          "dataset has held-out episodes: norm_stat fitted over data "
                          "the model never sees leaks the held-out action range into "
                          "the normalization, and the clip bounds it produces are not "
                          "the ones the training distribution implies. Must match the "
                          "`episode_subset` of the config that trains on this dataset.")
    ap.add_argument("--action-space", choices=("eef", "joint"), default="eef",
                     help="eef (default): FK to the 16-dim EEF layout, norm_stat on "
                          "segment-relative poses. joint: keep the raw 14-dim joints, "
                          "norm_stat on absolute angles in the base model's joint "
                          "slots (14:20, 21:27). Must match the training config's "
                          "action_space.")
    ap.add_argument("--latents-from", default=None, metavar="DIR",
                     help="Copy latents/ and empty_emb.pt from an existing "
                          "conversion of the same --src (e.g. the EEF one), after "
                          "checking its action_config segments are identical. Copied, "
                          "not hardlinked, so re-extracting either dataset's latents "
                          "cannot change the other's.")
    args = ap.parse_args()

    src, dst = args.src, args.dst
    joint = args.action_space == "joint"
    if os.path.exists(dst) and args.overwrite:
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "data", "chunk-000"), exist_ok=True)
    os.makedirs(os.path.join(dst, "meta"), exist_ok=True)

    fk = None if joint else build_fk(args.i2rt_root)

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

    norm_stat_episodes = parse_episode_spec(args.norm_stat_episodes)
    if norm_stat_episodes is not None:
        present = {e["episode_index"] for e in src_episodes}
        missing = sorted(norm_stat_episodes - present)
        assert not missing, f"--norm-stat-episodes names absent episodes: {missing}"
        fit = sorted(norm_stat_episodes)
        print(f"[norm_stat] fitting q01/q99 on {len(fit)} of {len(present)} episodes "
              f"({fit[0]}..{fit[-1]}); the rest are converted but held out of the fit")

    new_episodes = []
    new_stats_lines = []
    all_rel_actions = []

    for ep in src_episodes:
        idx = ep["episode_index"]
        length = ep["length"]
        fname = f"episode_{idx:06d}.parquet"
        src_table = pq.read_table(os.path.join(src, "data", "chunk-000", fname))
        assert src_table.num_rows == length

        state14 = read_list_column(src_table, "observation.state")
        action14 = read_list_column(src_table, "action")
        if joint:
            state_out, action_out = state14, action14
        else:
            state_out = joints14_to_eef16(state14, fk)
            action_out = joints14_to_eef16(action14, fk)

        write_parquet(os.path.join(dst, "data", "chunk-000", fname), src_table,
                      state_out, action_out)

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
            # what _action_post_process applies before normalization. Joint actions
            # reach normalization untransformed, so they are fitted as they are.
            if norm_stat_episodes is None or idx in norm_stat_episodes:
                seg = action_out[c_start:c_end]
                all_rel_actions.append(seg if joint else relativize_action16(seg))

        new_episodes.append({
            "episode_index": idx,
            "tasks": ep["tasks"],
            "length": length,
            "action_config": action_config,
        })

        src_stats = src_stats_by_ep[idx]["stats"]
        new_stats = dict(src_stats)
        new_stats["observation.state"] = col_stats(state_out)
        new_stats["action"] = col_stats(action_out)
        new_stats_lines.append({"episode_index": idx, "stats": new_stats})

        print(f"[convert] episode {idx:06d} ({length} frames, {len(chunks)} chunks) done")

    # 3. meta/info.json -- update state/action feature blocks to 16-dim EEF layout.
    #    Joint mode keeps --src's 14-dim blocks, which already describe its columns.
    new_info = deepcopy(src_info)
    for key in (() if joint else ("observation.state", "action")):
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
    #    already normalized [0,1] like robotwin's). Joint mode: absolute angles, raw
    #    [left q6, left grip, right q6, right grip] scattered into the joint slots.
    all_rel = np.concatenate(all_rel_actions, axis=0)  # (total_frames, D), raw order
    q01_raw = np.percentile(all_rel, 1, axis=0)
    q99_raw = np.percentile(all_rel, 99, axis=0)

    q01_30 = np.zeros(30)
    q99_30 = np.zeros(30)
    if joint:
        layout = ((LEFT_JOINT_CANON, slice(0, 6)), (LEFT_GRIPPER_CANON, 6),
                  (RIGHT_JOINT_CANON, slice(7, 13)), (RIGHT_GRIPPER_CANON, 13))
    else:
        layout = ((LEFT_EEF_CANON, slice(0, 7)), (LEFT_GRIPPER_CANON, 7),
                  (RIGHT_EEF_CANON, slice(8, 15)), (RIGHT_GRIPPER_CANON, 15))
    for canon, raw in layout:
        q01_30[canon] = q01_raw[raw]
        q99_30[canon] = q99_raw[raw]

    norm_stat = {"q01": q01_30.tolist(), "q99": q99_30.tolist()}
    # Record the fit set: which episodes these quantiles came from is not recoverable
    # from the numbers, and a norm_stat fitted on a different split than the one
    # trained is a mismatch nothing downstream can detect.
    if norm_stat_episodes is not None:
        norm_stat["fit_episodes"] = sorted(norm_stat_episodes)
    with open(os.path.join(dst, "norm_stat.json"), "w") as f:
        json.dump(norm_stat, f, indent=2)

    print("\n=== norm_stat (paste into the serving config of this dataset, e.g. "
          "wan_va/configs/va_yam_place_cube_cfg.py) ===")
    print("norm_stat = {")
    print(f"    'q01': {q01_30.tolist()},")
    print(f"    'q99': {q99_30.tolist()},")
    print("}")

    if args.latents_from:
        copy_latents(args.latents_from, dst, new_episodes, video_cameras(new_info))
    print(f"\nDone. Converted dataset written to: {dst}")
    if not args.latents_from:
        print("Next: copy empty_emb.pt from Datasets/robotwin-clean-and-aug-lerobot/, "
              "then run script/extract_va_latents.py to produce latents/.")


if __name__ == "__main__":
    main()
