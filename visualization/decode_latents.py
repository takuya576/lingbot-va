"""Decode saved server-mode latents into an MP4 of the imagined video.

Reads `latents_*.pt` files written by `VA_Server._infer` under
`visualization/real/<prompt>_<timestamp>/` and runs them through the same
VAE (with the same mean/std denormalization) used by
`wan_va.wan_va_server.VA_Server.decode_one_video`.
"""

import argparse
import os
import re
import sys
from glob import glob

import torch
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "wan_va"))

from modules.utils import load_vae  # noqa: E402

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def discover_latent_files(rollout_dir):
    pattern = os.path.join(rollout_dir, "latents_*.pt")
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No latents_*.pt files found in {rollout_dir}")

    def frame_st_id(path):
        m = re.search(r"latents_(\d+)\.pt$", os.path.basename(path))
        if m is None:
            raise ValueError(f"Cannot parse frame_st_id from {path}")
        return int(m.group(1))

    return sorted(files, key=frame_st_id)


def decode(latents, vae, video_processor):
    latents = latents.to(vae.dtype)
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    video = vae.decode(latents, return_dict=False)[0]
    return video_processor.postprocess_video(video, output_type="np")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_dir", required=True)
    parser.add_argument(
        "--vae_path",
        default=os.path.join(
            REPO_ROOT, "Pretrained_models/lingbot-va-posttrain-robotwin/vae"
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    output_path = args.output or os.path.join(args.rollout_dir, "imagined.mp4")

    latent_files = discover_latent_files(args.rollout_dir)
    print(f"Found {len(latent_files)} latent chunks:")
    for f in latent_files:
        print(f"  {os.path.basename(f)}")

    chunks = [torch.load(f, map_location=device) for f in latent_files]
    latents = torch.cat(chunks, dim=2)
    print(f"Concatenated latents shape: {tuple(latents.shape)}")

    vae = load_vae(args.vae_path, torch_dtype=dtype, torch_device=device)
    video_processor = VideoProcessor(vae_scale_factor=1)

    with torch.no_grad():
        frames = decode(latents, vae, video_processor)

    print(f"Decoded video shape: {frames.shape}")
    export_to_video(frames, output_path, fps=args.fps)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
