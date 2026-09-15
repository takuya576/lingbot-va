#!/usr/bin/env python3
"""Extract Wan2.2 VAE video latents + T5 text embeddings for a converted LingBot-VA
dataset (i.e. one that already has meta/episodes.jsonl action_config segments, as
produced by script/convert_yam_to_lingbot_va.py).

Conventions replicated here were verified empirically against the existing
Datasets/robotwin-clean-and-aug-lerobot cached latents (re-encoding a known robotwin
video and comparing against its stored .pth gave mean abs diff ~0.03 on a [-6,6]
range for video latents, and ~4e-5 for text embeddings -- i.e. this matches the
convention the training dataloader (wan_va/dataset/lerobot_latent_dataset.py) and the
live server (wan_va/wan_va_server.py's _encode_obs) both assume):
  - pixels: uint8 [0,255] -> float32, /255*2-1 -> [-1,1]
  - resize: torch bilinear, align_corners=False; for env_type == 'robotwin_tshape',
    camera index 0 ("cam_high") is resized to (height,width), all other cameras
    (wrist cams) are resized to (height//2, width//2)
  - encode: vae.encode(x).latent_dist.mode() (the raw posterior mean "mu")
  - normalize: (mu - vae.config.latents_mean) / vae.config.latents_std, per-channel
  - stored flattened as '(f h w) c' -> this is what's cached to latents/*.pth

Run in the `lingbot-va` conda env (needs torch/diffusers/transformers + a GPU).

Usage:
    conda activate lingbot-va
    python script/extract_va_latents.py \
        --dataset-path Datasets/yam-real-world-lingbot-va/cover_blocks \
        --config-name yam_lora_train
"""
import argparse
import json
import os
import sys

import av
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wan_va"))

from configs import VA_CONFIGS  # noqa: E402
from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: E402
from modules.utils import load_text_encoder, load_tokenizer, load_vae  # noqa: E402


def read_video_frames(path: str) -> np.ndarray:
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return np.stack(frames)  # (T,H,W,3) uint8


def pick_frame_ids(start_frame: int, end_frame: int, stride: int) -> list:
    """Largest (4k+1)-length, evenly-strided sample fitting in [start_frame, end_frame).

    The Wan2.2 causal VAE's temporal downsample expects video_num_frames == 4k+1 (see
    AutoencoderKLWan._encode: iter_ = 1 + (num_frame-1)//4, matching how the existing
    robotwin cached latents were produced: 139 raw frames, stride 4 -> 33 = 4*8+1).
    """
    length = end_frame - start_frame
    m = (length - 1) // stride + 1
    k = (m - 1) // 4
    n = 4 * k + 1
    return [start_frame + i * stride for i in range(n)]


def encode_video(vae, frames: np.ndarray, frame_ids: list, height: int, width: int,
                  device, dtype) -> torch.Tensor:
    sampled = frames[frame_ids].astype(np.float32)  # (F,H,W,3)
    vid = torch.from_numpy(sampled).permute(3, 0, 1, 2)  # (C,F,H,W)
    vid = F.interpolate(vid, size=(height, width), mode="bilinear", align_corners=False)
    vid = vid.unsqueeze(0) / 255.0 * 2.0 - 1.0  # (1,C,F,h,w)
    vid = vid.to(device=device, dtype=dtype)
    with torch.no_grad():
        mu = vae.encode(vid).latent_dist.mode()  # (1,z,f,h,w)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32)
    latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32)
    mu_norm = (mu.float() - latents_mean.view(1, -1, 1, 1, 1)) / latents_std.view(1, -1, 1, 1, 1)
    return mu_norm[0].to(dtype)  # (z,f,h,w)


def get_text_emb(tokenizer, text_encoder, prompt: str, device, dtype,
                  max_sequence_length: int = 512) -> torch.Tensor:
    prompt = prompt_clean(prompt)
    text_inputs = tokenizer([prompt], padding="max_length", max_length=max_sequence_length,
                             truncation=True, add_special_tokens=True,
                             return_attention_mask=True, return_tensors="pt")
    input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_len = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        out = text_encoder(input_ids.to(device), mask.to(device)).last_hidden_state
    out = out.to(dtype)
    u = out[0][:seq_len[0]]
    padded = torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
    return padded.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True,
                     help="Path to the converted per-task dataset, e.g. "
                          "Datasets/yam-real-world-lingbot-va/cover_blocks")
    ap.add_argument("--config-name", default="yam_lora_train",
                     help="wan_va config to source wan22_pretrained_model_name_or_path, "
                          "height/width, obs_cam_keys, env_type from")
    ap.add_argument("--target-fps", type=int, default=5,
                     help="Sampling fps for latent extraction; the actual frame stride "
                          "is round(ori_fps/target_fps), so 5 on 30fps source = stride "
                          "6. This is the lever that trades temporal resolution against "
                          "how much wall-clock duration fits in max_seq_len: packed "
                          "tokens = (k+1)*(240+8*stride). 5fps (the low end of the "
                          "5-15fps range the repo README sanctions) is what lets a "
                          "whole 40-70s cover_blocks episode fit in one segment, so "
                          "every training sample is anchored at episode frame 0 -- the "
                          "same anchor the deployment client holds for a whole rollout.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = VA_CONFIGS[args.config_name]
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    model_root = cfg.wan22_pretrained_model_name_or_path
    vae = load_vae(os.path.join(model_root, "vae"), torch_dtype=dtype, torch_device=device)
    tokenizer = load_tokenizer(os.path.join(model_root, "tokenizer"))
    text_encoder = load_text_encoder(os.path.join(model_root, "text_encoder"),
                                      torch_dtype=dtype, torch_device=device)

    obs_cam_keys = cfg.obs_cam_keys
    height, width = cfg.height, cfg.width
    is_tshape = cfg.env_type == "robotwin_tshape"

    with open(os.path.join(args.dataset_path, "meta", "info.json")) as f:
        info = json.load(f)
    ori_fps = info["fps"]
    stride = max(1, round(ori_fps / args.target_fps))

    with open(os.path.join(args.dataset_path, "meta", "episodes.jsonl")) as f:
        episodes = [json.loads(l) for l in f]

    text_emb_cache = {}

    for ep in episodes:
        idx = ep["episode_index"]
        for seg in ep["action_config"]:
            start_frame, end_frame = seg["start_frame"], seg["end_frame"]
            action_text = seg["action_text"]

            if action_text not in text_emb_cache:
                text_emb_cache[action_text] = get_text_emb(tokenizer, text_encoder, action_text,
                                                            device, dtype)
            text_emb = text_emb_cache[action_text]

            frame_ids = pick_frame_ids(start_frame, end_frame, stride)
            video_num_frames = len(frame_ids)

            for cam_i, cam_key in enumerate(obs_cam_keys):
                out_dir = os.path.join(args.dataset_path, "latents", "chunk-000", cam_key)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"episode_{idx:06d}_{start_frame}_{end_frame}.pth")
                if os.path.exists(out_path) and not args.overwrite:
                    continue

                if is_tshape and cam_i > 0:
                    cam_h, cam_w = height // 2, width // 2
                else:
                    cam_h, cam_w = height, width

                video_path = os.path.join(args.dataset_path, "videos", "chunk-000", cam_key,
                                           f"episode_{idx:06d}.mp4")
                frames = read_video_frames(video_path)
                latent = encode_video(vae, frames, frame_ids, cam_h, cam_w, device, dtype)
                z, f, h, w = latent.shape
                latent_flat = rearrange(latent, "c f h w -> (f h w) c").cpu()

                torch.save({
                    "latent": latent_flat,
                    "latent_num_frames": f,
                    "latent_height": h,
                    "latent_width": w,
                    "video_num_frames": video_num_frames,
                    "video_height": cam_h,
                    "video_width": cam_w,
                    "text_emb": text_emb,
                    "text": action_text,
                    "frame_ids": frame_ids,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": args.target_fps,
                    "ori_fps": ori_fps,
                }, out_path)

            print(f"[latents] episode {idx:06d} segment [{start_frame},{end_frame}) "
                  f"-> {video_num_frames} frames done")

    print("Done.")


if __name__ == "__main__":
    main()
