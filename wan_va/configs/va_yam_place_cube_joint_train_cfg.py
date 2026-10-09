# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Post-training config for YAM `place_cube` on JOINT-space actions, from `lingbot-va-base`.

The joint-space twin of `va_yam_place_cube_base_train_cfg`: same initialization,
episode split, LoRA shape, optimizer, steps and max_seq_len, so the action space is
the only variable between the two runs. Keep the two in step when changing either.

Inherits from `va_yam_place_cube_joint_cfg`, the config that serves this run's
checkpoints: action_space, the joint channel layout, norm_stat, action_per_frame and
the camera setup all come from there, so training and serving cannot drift apart.
Only the initialization, the dataset and the optimization settings are
training-specific.

max_seq_len is the EEF run's: packed length depends on the latent frame count and
action_per_frame, not on how many action channels are used.

`full_train_action_io` = True matters more here than for the EEF runs: the joint
slots' action_embedder/action_proj_out rows were last fitted to the pretraining
mixture's joint quantiles, and they are now fed YAM's.
"""
import os

import torch
from easydict import EasyDict

from .va_yam_place_cube_joint_cfg import va_yam_place_cube_joint_cfg

va_yam_place_cube_joint_train_cfg = EasyDict(
    __name__='Config: VA yam place_cube train (joint, base init)')
va_yam_place_cube_joint_train_cfg.update(va_yam_place_cube_joint_cfg)

# The un-adapted base, not va_yam_place_cube_joint_cfg's serving directory (that is
# this run's output).
va_yam_place_cube_joint_train_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-base')

# The joint conversion: same episodes, a copy of the EEF conversion's latents, 14-dim
# joint parquet.
va_yam_place_cube_joint_train_cfg.dataset_path = (
    '/home/takuya/lingbot-va/Datasets/yam-real-world-lingbot-va-joint/place_cube')
va_yam_place_cube_joint_train_cfg.empty_emb_path = os.path.join(
    va_yam_place_cube_joint_train_cfg.dataset_path, 'empty_emb.pt')
# Train/val split: episodes 0-49 train, 50-54 held out, as in the EEF place_cube runs.
va_yam_place_cube_joint_train_cfg.episode_subset = list(range(50))
# Separate from train_out/yam_place_cube_base, so assemble_yam_serving_model.sh can
# never assemble an EEF checkpoint into the joint serving directory or the reverse --
# serving either under the other's config denormalizes into the wrong channels.
va_yam_place_cube_joint_train_cfg.save_root = './train_out/yam_place_cube_joint'
va_yam_place_cube_joint_train_cfg.enable_wandb = True
va_yam_place_cube_joint_train_cfg.load_worker = 16
va_yam_place_cube_joint_train_cfg.save_interval = 1000
va_yam_place_cube_joint_train_cfg.gc_interval = 50
va_yam_place_cube_joint_train_cfg.cfg_prob = 0.1

# LoRA
va_yam_place_cube_joint_train_cfg.use_lora = True
va_yam_place_cube_joint_train_cfg.lora_rank = 32
va_yam_place_cube_joint_train_cfg.lora_alpha = 64
va_yam_place_cube_joint_train_cfg.lora_dropout = 0.0
va_yam_place_cube_joint_train_cfg.full_train_action_io = True
va_yam_place_cube_joint_train_cfg.full_train_action_cond_embedder = False
# Nothing updates the frozen base, so fp32 master copies of it are wasted:
# bf16 halves the base footprint from 10.2GB to 5.1GB per rank at world_size=2.
va_yam_place_cube_joint_train_cfg.base_param_dtype = torch.bfloat16

# Training parameters.
va_yam_place_cube_joint_train_cfg.learning_rate = 1e-5
va_yam_place_cube_joint_train_cfg.beta1 = 0.9
va_yam_place_cube_joint_train_cfg.beta2 = 0.95
va_yam_place_cube_joint_train_cfg.weight_decay = 0.0
va_yam_place_cube_joint_train_cfg.warmup_steps = 20
va_yam_place_cube_joint_train_cfg.batch_size = 1
va_yam_place_cube_joint_train_cfg.gradient_accumulation_steps = 16
va_yam_place_cube_joint_train_cfg.num_steps = 3000

# Admits every place_cube episode at stride 4 (longest: 11520 tokens, episode 17);
# see va_yam_place_cube_train_cfg for the derivation.
va_yam_place_cube_joint_train_cfg.max_seq_len = 11520
