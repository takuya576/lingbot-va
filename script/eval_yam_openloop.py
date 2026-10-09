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

Caveat, for cover_blocks (the --src/--dst defaults): all 74 episodes were trained on --
there is no held-out set. A good result rules out gross failure (unlearned adapter, wrong
norm stats, broken observation pipeline). It is not a success-rate estimate.

place_cube DOES have a held-out set: va_yam_place_cube_train_cfg.episode_subset
trains on episodes 0-49 only, so evaluating 50-54 is a genuine generalization test. The
held-out episodes are skipped when building the training metas but remain fully present
on disk, so nothing here needs to change to read them. (place_cube_pre, the original
57-episode recording, holds out 50-56 instead: serve CONFIG_NAME=yam_place_cube_pre
and pass its own --src/--dst.)

Usage (needs the policy server up:
NGPU=1 CONFIG_NAME=yam_cover_blocks bash script/run_launch_va_server_sync.sh)::

    /data2/takuya/raiden/.venv/bin/python script/eval_yam_openloop.py --episodes 2

    # place_cube held-out set (server launched with CONFIG_NAME=yam_place_cube or
    # yam_place_cube_base)
    /data2/takuya/raiden/.venv/bin/python script/eval_yam_openloop.py \
        --src Datasets/yam-real-world-lerobot/place_cube \
        --dst Datasets/yam-real-world-lingbot-va/place_cube \
        --episodes 50 51 52 53 54

Output goes to the checkpoint's own position mirrored under eval_out/, e.g.
eval_out/yam_place_cube/<run>/checkpoints/checkpoint_step_N/open_loop/ -- the leaf
directory is what separates this from a closed-loop evaluation of the same checkpoint.
The checkpoint is the one the server reports it loaded, not one named here, so the
results cannot be filed under a model other than the one that produced them;
--config-name and --checkpoint only stop the run if they disagree with that report.
--out overrides the location, not the check.

--dst must be the dataset the served model was trained on: its norm_stat.json has to
match the quantiles the server denormalizes with, which ties it to exactly one config.
A joint-space checkpoint (CONFIG_NAME=yam_place_cube_joint) therefore takes
--dst Datasets/yam-real-world-lingbot-va-joint/place_cube. It is scored in the same
EEF frame as the EEF checkpoints -- ground truth by FK of the recorded joints, which
is exactly what the EEF dataset holds -- plus a joint-space error, which is what that
model was actually trained on.
--src/--dst default to cover_blocks, so a place_cube server stops here until they are
passed.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yam_bridge.action_convert import (  # noqa: E402
    DEFAULT_I2RT_ROOT, YAM_DOF, YamActionConverter, joints14_to_eef16, make_converter,
    relativize_action16,
)
from yam_bridge.eval_meta import (  # noqa: E402
    artifact_stem, eval_out_dir, resolve_served,
)
from yam_bridge.eval_viz import (  # noqa: E402
    ARMS, CAM_KEYS, plot_trajectories, rotation_error_deg, save_imagination,
)
from yam_bridge.kinematics import ARM_DOF  # noqa: E402
from yam_bridge.va_client import VAClient  # noqa: E402

ARM_JOINTS = np.r_[0:ARM_DOF, YAM_DOF:YAM_DOF + ARM_DOF]


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


def evaluate(ep: int, args, va: VAClient, conv: YamActionConverter) -> dict:
    src = os.path.join(args.src, "data", "chunk-000", f"episode_{ep:06d}.parquet")
    dst = os.path.join(args.dst, "data", "chunk-000", f"episode_{ep:06d}.parquet")
    joints14 = read_column(src, "action")
    if conv.action_space == "eef":
        gt_rel = relativize_action16(read_column(dst, "action"))
    else:
        # The joint dataset's actions are --src's joints, so FK them the way the EEF
        # conversion did; the EEF scores then mean the same for both kinds of model.
        gt_rel = relativize_action16(np.stack(
            [joints14_to_eef16(q, conv.kin_left, conv.kin_right) for q in joints14]))

    with open(os.path.join(args.dst, "meta", "episodes.jsonl")) as f:
        meta = {json.loads(l)["episode_index"]: json.loads(l) for l in f}
    prompt = meta[ep]["tasks"][0]

    # Chunk 0 executes only latent frame 1 (action_per_frame steps); later chunks
    # execute two. A keyframe follows every action_per_frame/4 steps -- the causal VAE
    # consumes 4 frames per latent frame -- which is the 5 Hz the latents were extracted
    # at. All of it scales with action_per_frame: 24/48/6 for cover_blocks at stride 6,
    # 16/32/4 for place_cube at stride 4. Hardcoding the cover_blocks numbers used to
    # hand va.advance a keyframe list of the wrong length on any other task.
    apf = args.action_per_frame
    stride = apf // 4
    n_chunks = args.max_chunks
    total = apf + 2 * apf * (n_chunks - 1)
    if total + 8 >= len(joints14):
        n_chunks = max(1, (len(joints14) - (apf + 8)) // (2 * apf))
        total = apf + 2 * apf * (n_chunks - 1)
    keyframe_steps = list(range(stride, total + 1, stride))
    frames = decode_frames_all(args.dst, ep, set([0] + keyframe_steps))

    va.reset(prompt)
    conv.reset_stats()
    conv.latch_anchor(joints14[0])

    first_obs = frames[0]
    preds, cmds_all, bounds, videos = [], [], [], []
    pred_joints = []   # joint checkpoints only: raw predicted joints, before clipping
    step = 0
    t_total = 0.0
    t_video = 0.0
    t_vgen = 0.0
    t_act = 0.0

    for c in range(n_chunks):
        t0 = time.perf_counter()
        got = va.infer(first_obs, want_video=args.video)
        action, imagined = got if args.video else (got, None)
        # The server decodes the imagined video inside this blocking call, so the round
        # trip is not inference once --video is on. Timed separately, or turning the
        # visualisation on would silently multiply the reported per-chunk latency.
        t_video += va.last_video_decode_ms / 1e3
        t_vgen += va.last_video_gen_ms / 1e3
        t_act += va.last_action_gen_ms / 1e3
        t_total += time.perf_counter() - t0 - va.last_video_decode_ms / 1e3

        # The identity the results are filed under only predicts what the server
        # does; this is what it did. A disagreement means that identity is wrong --
        # possible only against a server too old to report its own, where it came from
        # an unverified --config-name -- so stop rather than file under the wrong model.
        if action.shape[2] != apf:
            raise SystemExit(
                f"server returned action_per_frame={action.shape[2]}, but "
                f"{args.config_name} says {apf}. Re-run with the CONFIG_NAME the "
                f"policy server was launched with.")

        start_idx = 1 if c == 0 else 0
        executed = action[:, start_idx:, :]
        bounds.append(step)   # cumulative control steps before this chunk
        preds.append(conv.predicted_rel16(executed))
        cmds_all.append(conv.convert_chunk(executed))
        if conv.action_space == "joint":
            pred_joints.append(executed.transpose(1, 2, 0).reshape(-1, YAM_DOF * 2))

        n_exec = preds[-1].shape[0]
        ks = [step + s for s in range(stride, n_exec + 1, stride)]
        step += n_exec
        # frames is {frame_index: {cam_key: array}}, which is already the per-timestep
        # dict shape the server expects.
        va.advance([frames[idx] for idx in ks], action)
        # The server decodes the real latents in here too when video is on, to
        # keep the imagined video continuing from reality. Part of what --video
        # costs, so it belongs with the other decode, not unreported.
        t_video += va.last_real_decode_ms / 1e3

        if args.video:
            videos.append((imagined, [frames[k] for k in ks]))

    pred = np.concatenate(preds)
    cmds = np.concatenate(cmds_all)
    # Executed action j corresponds to ground-truth frame j. Training pads
    # frame_stride*4 = 24 zeros at the *front* before reshaping into latent frames
    # (lerobot_latent_dataset.py:268), so latent frame 0 holds the pad -- which is why
    # it is skipped -- and latent frame i covers raw frames (i-1)*24 .. i*24-1. Chunk 0's
    # single executed frame is therefore raw frames 0..23, not 1..24.
    # Executed action j is ground-truth frame j -- see the padding note above. Plotted
    # and scored at that alignment and no other: shifting to whatever offset minimises
    # error would flatter the result and hide a genuine timing lead or lag in the policy.
    gt = gt_rel[:len(pred)]
    n = min(len(gt), len(pred))
    pred, gt = pred[:n], gt[:n]

    res = {"episode": ep, "chunks": n_chunks, "steps": n,
           "infer_ms": t_total / n_chunks * 1e3,
           "video_gen_ms": t_vgen / n_chunks * 1e3,
           "action_gen_ms": t_act / n_chunks * 1e3,
           "video_decode_ms": t_video / n_chunks * 1e3,
           "stats": dict(conv.stats), "arms": {},
           # The filename tag carries only the step counts.
           "served": args.served_settings}
    for arm, (xyz, quat, grip) in ARMS.items():
        terr = np.linalg.norm(pred[:, xyz] - gt[:, xyz], axis=1) * 1000
        rerr = rotation_error_deg(pred[:, quat], gt[:, quat])
        res["arms"][arm] = {
            "trans_mm": np.percentile(terr, [50, 99]).tolist() + [terr.max()],
            "rot_deg": np.percentile(rerr, [50, 99]).tolist() + [rerr.max()],
            "grip_mae": float(np.abs(pred[:, grip] - gt[:, grip]).mean()),
        }
    res["max_step_delta"] = float(np.abs(np.diff(cmds[:, ARM_JOINTS], axis=0)).max())
    if pred_joints:
        # Same alignment as the EEF scores: executed action j is ground-truth frame j.
        jerr = np.abs(np.concatenate(pred_joints)[:n] - joints14[:n])
        res["joint_err_rad"] = {
            arm: np.percentile(jerr[:, sl].max(axis=1), [50, 99]).tolist()
            + [float(jerr[:, sl].max())]
            for arm, sl in (("left", slice(0, ARM_DOF)),
                            ("right", slice(YAM_DOF, YAM_DOF + ARM_DOF)))}

    os.makedirs(args.out, exist_ok=True)
    tag = f"episode_{ep:06d}"
    # Filenames carry the inference settings; the plot title stays readable.
    stem = artifact_stem(ep, args.param_tag)
    plot_trajectories(
        [{"name": "ground truth", "data": gt, "style": {"ls": "-", "lw": 1.6}},
         {"name": "policy", "data": pred, "style": {"ls": "--", "lw": 1.4}}],
        bounds, os.path.join(args.out, f"{stem}_traj.png"), tag)
    if args.video and videos:
        save_imagination(videos, os.path.join(args.out, f"{stem}_imagined.mp4"),
                         labels=args.labels)
    return res


def decode_frames_all(dataset_root: str, ep: int, wanted: set) -> Dict[int, Dict[str, np.ndarray]]:
    """{frame_index: {cam_key: RGB array}} for the three cameras."""
    per_cam = {k: decode_frames(os.path.join(dataset_root, "videos", "chunk-000", k),
                                ep, wanted) for k in CAM_KEYS}
    return {i: {k: per_cam[k][i] for k in CAM_KEYS} for i in wanted}


def check_dataset(dst: str, served: dict) -> None:
    """Stop unless `dst` is the dataset the served model's quantiles were fitted on.

    Nothing else would notice. The protocol runs the same on any recorded episode, and
    the action_per_frame check in `evaluate` only separates strides, so a place_cube
    model scored against cover_blocks episodes yields ordinary-looking numbers measuring
    nothing. Each config's norm_stat matches exactly one dataset's norm_stat.json.
    """
    path = os.path.join(dst, "norm_stat.json")
    with open(path) as f:
        ds = json.load(f)
    if not all(np.allclose(np.asarray(ds[k], dtype=np.float64),
                           np.asarray(served["norm_stat"][k], dtype=np.float64),
                           rtol=0, atol=1e-9) for k in ("q01", "q99")):
        raise SystemExit(
            f"{path} does not match the norm_stat of {served['config_name']}, the "
            f"config being served: --dst is not the dataset this model was trained on. "
            f"Pass the matching --src/--dst.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Datasets/yam-real-world-lerobot/cover_blocks")
    ap.add_argument("--dst", default="Datasets/yam-real-world-lingbot-va/cover_blocks")
    ap.add_argument("--i2rt-root", default=DEFAULT_I2RT_ROOT)
    ap.add_argument("--va-host", default="0.0.0.0")
    ap.add_argument("--va-port", type=int, default=29536)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1], metavar="N",
                    help="Episode indices to evaluate, e.g. --episodes 0 5 42. "
                         "Not a count: these are the episode numbers themselves.")
    ap.add_argument("--max-chunks", type=int, default=20)
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
                    help="Output directory. Defaults to the checkpoint's position "
                         "mirrored under eval_out/, plus open_loop/.")
    ap.add_argument("--no-video", dest="video", action="store_false",
                    help="Skip the imagined-video dump (saves a server-side VAE decode).")
    ap.add_argument("--no-labels", dest="labels", action="store_false",
                    help="Render the video without the 'imagined'/'ground truth' captions "
                         "and the chunk/frame counters (on by default).")
    args = ap.parse_args()

    available = {int(f[len("episode_"):-len(".parquet")])
                 for f in os.listdir(os.path.join(args.src, "data", "chunk-000"))
                 if f.startswith("episode_") and f.endswith(".parquet")}
    missing = sorted(set(args.episodes) - available)
    if missing:
        raise SystemExit(
            f"no such episode(s): {missing}. {len(available)} available, "
            f"{min(available)}..{max(available)}")

    # Checked before connecting: VAClient blocks indefinitely waiting for the server, so
    # a bad episode index should surface immediately rather than after that hang.
    print(f"connecting to policy server {args.va_host}:{args.va_port} "
          "(blocks until it accepts)...")
    va = VAClient(host=args.va_host, port=args.va_port)

    # Filed under what the server says it loaded; --config-name and --checkpoint only
    # stop the run if they disagree with it.
    served = resolve_served(va.server_metadata, args.config_name, args.checkpoint)
    args.config_name = served['config_name']
    args.param_tag = served['param_tag']
    args.action_per_frame = served['action_per_frame']
    args.served_settings = served['settings']
    print(f"[va] settings {served['settings']}")
    print(f"[va] serving {served['config_name']} from "
          f"{served['checkpoint'] or '<unlinked serving directory>'}")
    check_dataset(args.dst, served)
    if args.out is None:
        args.out = eval_out_dir('open_loop', served['checkpoint'])
    print(f"writing to {args.out}/ as ep<N>_{args.param_tag}_*")
    conv = make_converter(served['action_space'], args.i2rt_root)
    print(f"[va] action space: {served['action_space']}")

    results = []
    for ep in args.episodes:
        print(f"\n=== episode {ep} ===")
        r = evaluate(ep, args, va, conv)
        results.append(r)
        s = r["stats"]
        vid = (f" + {r['video_decode_ms']:.0f} ms/chunk video decode"
               if r.get("video_decode_ms") else "")
        print(f"  {r['steps']} steps over {r['chunks']} chunks, "
              f"{r['infer_ms']:.0f} ms/chunk inference "
              f"(video latents {r['video_gen_ms']:.0f} + actions "
              f"{r['action_gen_ms']:.0f}){vid}")
        for arm, m in r["arms"].items():
            print(f"  {arm:5s} trans p50/p99/max {m['trans_mm'][0]:7.1f}/"
                  f"{m['trans_mm'][1]:7.1f}/{m['trans_mm'][2]:7.1f} mm   "
                  f"rot {m['rot_deg'][0]:5.1f}/{m['rot_deg'][1]:5.1f}/{m['rot_deg'][2]:5.1f} deg   "
                  f"grip MAE {m['grip_mae']:.3f}")
        if conv.action_space == "eef":
            print(f"  IK on model outputs: {s['ik_failures']} failures, "
                  f"{s['ik_fallbacks']} fallbacks, {s['unsafe_jumps']} unsafe jumps, "
                  f"max residual {s['max_pos_residual_m'] * 1000:.2f} mm")
        else:
            for arm, e in r["joint_err_rad"].items():
                print(f"  {arm:5s} joint err (worst joint) p50/p99/max "
                      f"{e[0]:.4f}/{e[1]:.4f}/{e[2]:.4f} rad")
            print(f"  joint limits: {s['joint_limit_clipped']}/{s['steps']} steps "
                  f"clipped (worst overshoot {s['max_joint_overshoot_rad']:.3f} rad)")
        print(f"  max per-step joint delta: {r['max_step_delta']:.4f} rad "
              f"(sets --max-joint-delta)")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote plots, video and results.json to {args.out}/")


if __name__ == "__main__":
    main()
