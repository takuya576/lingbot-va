# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Post-training config for the YAM `place_cube` task -- FIRST recording (`place_cube_pre`).

Kept for reproducing and resuming the runs trained on the original 57-episode
recording; `va_yam_place_cube_train_cfg` trains on the re-recorded 55-episode dataset.

Mirrors `va_yam_place_cube_train_cfg` (see there for why LoRA rather than a full
finetune, and for the weight-decay/warmup rationale). What differs is the dataset and
its held-out split.

Inherits from `va_yam_place_cube_pre_cfg`, the config that serves this run's
checkpoints: action_per_frame, norm_stat, the EEF channel layout and the camera setup
all come from there, so training and serving cannot drift apart. Only the
initialization, the dataset and the optimization settings are training-specific.
"""
import os

import torch
from easydict import EasyDict

from .va_yam_place_cube_pre_cfg import va_yam_place_cube_pre_cfg

va_yam_place_cube_pre_train_cfg = EasyDict(
    __name__='Config: VA yam place_cube_pre train')
va_yam_place_cube_pre_train_cfg.update(va_yam_place_cube_pre_cfg)

# va_yam_place_cube_pre_cfg points this at the LoRA-merged serving directory, which is
# the *output* of this run. Training must start from the original un-adapted base
# instead -- LoRA injection asserts it (wan_va/modules/lora.py:223), and resuming
# against merged weights would fold the adapter in a second time.
va_yam_place_cube_pre_train_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-robotwin')

# The SINGLE task directory, not the yam-real-world-lingbot-va parent; see
# va_yam_place_cube_train_cfg for why.
va_yam_place_cube_pre_train_cfg.dataset_path = (
    '/home/takuya/lingbot-va/Datasets/yam-real-world-lingbot-va/place_cube_pre')
va_yam_place_cube_pre_train_cfg.empty_emb_path = os.path.join(
    va_yam_place_cube_pre_train_cfg.dataset_path, 'empty_emb.pt')
# Train/val split: episodes 0-49 train, 50-56 held out. There is no in-training
# validation loop in this repo, so the held-out set is scored offline against the
# running policy server:
#
#   python script/eval_yam_openloop.py --config-name yam_place_cube_pre \
#       --src Datasets/yam-real-world-lerobot/place_cube_pre \
#       --dst Datasets/yam-real-world-lingbot-va/place_cube_pre \
#       --episodes 50 51 52 53 54 55 56
#
# The held-out episodes keep their parquet rows, videos and cached latents on disk --
# they are only skipped when building new_metas -- which is what lets that command
# read them. norm_stat in va_yam_place_cube_pre_cfg is fitted on 0-49 to match; changing
# this split means refitting it.
va_yam_place_cube_pre_train_cfg.episode_subset = list(range(50))
va_yam_place_cube_pre_train_cfg.save_root = './train_out/yam_place_cube_pre'
va_yam_place_cube_pre_train_cfg.enable_wandb = True
va_yam_place_cube_pre_train_cfg.load_worker = 16
va_yam_place_cube_pre_train_cfg.save_interval = 1000
va_yam_place_cube_pre_train_cfg.gc_interval = 50
va_yam_place_cube_pre_train_cfg.cfg_prob = 0.1

# LoRA
va_yam_place_cube_pre_train_cfg.use_lora = True
va_yam_place_cube_pre_train_cfg.lora_rank = 32
va_yam_place_cube_pre_train_cfg.lora_alpha = 64
va_yam_place_cube_pre_train_cfg.lora_dropout = 0.0
va_yam_place_cube_pre_train_cfg.full_train_action_io = True
va_yam_place_cube_pre_train_cfg.full_train_action_cond_embedder = False
# Nothing updates the frozen base, so fp32 master copies of it are wasted:
# bf16 halves the base footprint from 10.2GB to 5.1GB per rank at world_size=2.
va_yam_place_cube_pre_train_cfg.base_param_dtype = torch.bfloat16

# Training parameters.
va_yam_place_cube_pre_train_cfg.learning_rate = 1e-5
va_yam_place_cube_pre_train_cfg.beta1 = 0.9
va_yam_place_cube_pre_train_cfg.beta2 = 0.95
va_yam_place_cube_pre_train_cfg.weight_decay = 0.0
va_yam_place_cube_pre_train_cfg.warmup_steps = 20
va_yam_place_cube_pre_train_cfg.batch_size = 1
va_yam_place_cube_pre_train_cfg.gradient_accumulation_steps = 16
va_yam_place_cube_pre_train_cfg.num_steps = 3000

# packed_tokens = (k+1)*(240+8*stride), rounded up to a multiple of 128
# (lerobot_latent_dataset.py:287-306). At stride 4 the 57 place_cube_pre episodes span
# 4624 tokens (259 frames) to 11152 (651 frames), so 11264 admits every one of them.
# Sizing this too small does not fail loudly: __getitem__ SKIPS to the next sample
# rather than asserting (lerobot_latent_dataset.py:308-320), so over-long episodes
# would just vanish from training while their neighbours got sampled twice.
va_yam_place_cube_pre_train_cfg.max_seq_len = 11264
