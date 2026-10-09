# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Post-training config for YAM `place_cube`, initialized from `lingbot-va-base`.

`va_yam_place_cube_train_cfg` starts from `lingbot-va-posttrain-robotwin`, which has
already been post-trained on RoboTwin. This one starts from the pre-trained
`lingbot-va-base` (robbyant/lingbot-va-base, rev 68b7bc1b) instead. EVERYTHING else --
dataset, episode split, norm_stat, LoRA shape, optimizer, steps, max_seq_len -- is
identical to `va_yam_place_cube_train_cfg`, so the initialization is the only variable
between the two runs. Keep the two in step when changing either.

Inherits from `va_yam_place_cube_base_cfg`, the config that serves this run's
checkpoints: action_per_frame, norm_stat, the EEF channel layout and the camera setup
all come from there, so training and serving cannot drift apart. Only the
initialization, the dataset and the optimization settings are training-specific.

The cached place_cube latents, text embeddings and empty_emb.pt are reused as-is. That
is only valid because the two checkpoints differ in `transformer/` alone: every file of
`vae/`, `text_encoder/` and `tokenizer/` has the same sha256 on both HF repos, and
Pretrained_models/lingbot-va-base hard-links them from the robotwin copy. A base whose
VAE or T5 differed would need latents and embeddings re-extracted.
"""
import os

import torch
from easydict import EasyDict

from .va_yam_place_cube_base_cfg import va_yam_place_cube_base_cfg

va_yam_place_cube_base_train_cfg = EasyDict(
    __name__='Config: VA yam place_cube train (base init)')
va_yam_place_cube_base_train_cfg.update(va_yam_place_cube_base_cfg)

# The un-adapted base, not the serving directory of va_yam_place_cube_base_cfg (that is
# this run's output). Its transformer/config.json says attn_mode "flex" where robotwin's
# says "torch"; neither is read, since train.py and wan_va_server.py both pass
# attn_mode explicitly to from_pretrained.
va_yam_place_cube_base_train_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-base')

# The SINGLE place_cube task directory; see va_yam_place_cube_train_cfg for why not
# the parent.
va_yam_place_cube_base_train_cfg.dataset_path = (
    '/home/takuya/lingbot-va/Datasets/yam-real-world-lingbot-va/place_cube')
va_yam_place_cube_base_train_cfg.empty_emb_path = os.path.join(
    va_yam_place_cube_base_train_cfg.dataset_path, 'empty_emb.pt')
# Train/val split: episodes 0-49 train, 50-54 held out, as in
# va_yam_place_cube_train_cfg (see there for the offline eval command).
va_yam_place_cube_base_train_cfg.episode_subset = list(range(50))
# Separate from train_out/yam_place_cube, so that assemble_yam_serving_model.sh's
# "newest checkpoint" default can never pick a base-initialized run for the
# robotwin-initialized serving directory, or the reverse.
va_yam_place_cube_base_train_cfg.save_root = './train_out/yam_place_cube_base'
va_yam_place_cube_base_train_cfg.enable_wandb = True
va_yam_place_cube_base_train_cfg.load_worker = 16
va_yam_place_cube_base_train_cfg.save_interval = 1000
va_yam_place_cube_base_train_cfg.gc_interval = 50
va_yam_place_cube_base_train_cfg.cfg_prob = 0.1

# LoRA
va_yam_place_cube_base_train_cfg.use_lora = True
va_yam_place_cube_base_train_cfg.lora_rank = 32
va_yam_place_cube_base_train_cfg.lora_alpha = 64
va_yam_place_cube_base_train_cfg.lora_dropout = 0.0
va_yam_place_cube_base_train_cfg.full_train_action_io = True
va_yam_place_cube_base_train_cfg.full_train_action_cond_embedder = False
# Nothing updates the frozen base, so fp32 master copies of it are wasted:
# bf16 halves the base footprint from 10.2GB to 5.1GB per rank at world_size=2.
va_yam_place_cube_base_train_cfg.base_param_dtype = torch.bfloat16

# Training parameters.
va_yam_place_cube_base_train_cfg.learning_rate = 1e-5
va_yam_place_cube_base_train_cfg.beta1 = 0.9
va_yam_place_cube_base_train_cfg.beta2 = 0.95
va_yam_place_cube_base_train_cfg.weight_decay = 0.0
va_yam_place_cube_base_train_cfg.warmup_steps = 20
va_yam_place_cube_base_train_cfg.batch_size = 1
va_yam_place_cube_base_train_cfg.gradient_accumulation_steps = 16
va_yam_place_cube_base_train_cfg.num_steps = 3000

# Admits every place_cube episode at stride 4 (longest: 11520 tokens, episode 17);
# see va_yam_place_cube_train_cfg for the derivation.
va_yam_place_cube_base_train_cfg.max_seq_len = 11520
