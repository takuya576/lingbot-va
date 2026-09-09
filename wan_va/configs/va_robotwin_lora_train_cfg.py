# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LoRA post-training config, for GPUs that cannot hold a full finetune.

Full finetuning of the 5.089B-parameter transformer peaks at 80-90GB/GPU, almost
all of it optimizer-side (fp32 grads + AdamW states + master weights). LoRA
removes that, bringing the run to roughly 22-32GB/GPU at world_size=2.

To retarget this at a different embodiment (e.g. a YAM arm fed by IK downstream
of the action decoder), copy this file and override:
  - `dataset_path` / `empty_emb_path`
  - `norm_stat`               recomputed q01/q99 from the new dataset
  - `used_action_channel_ids` the new EEF channel layout
  - `save_root`
`full_train_action_io` is already True, which is what lets the action heads
relearn a changed action convention.
"""
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os
import torch

va_robotwin_lora_train_cfg = EasyDict(__name__='Config: VA robotwin LoRA train')
va_robotwin_lora_train_cfg.update(va_robotwin_cfg)

va_robotwin_lora_train_cfg.dataset_path = '/home/takuya/lingbot-va/Datasets/robotwin-clean-and-aug-lerobot'
va_robotwin_lora_train_cfg.empty_emb_path = os.path.join(
    va_robotwin_lora_train_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_lora_train_cfg.save_root = './train_out/robotwin_lora'
va_robotwin_lora_train_cfg.enable_wandb = True
va_robotwin_lora_train_cfg.load_worker = 16
va_robotwin_lora_train_cfg.save_interval = 500
va_robotwin_lora_train_cfg.gc_interval = 50
va_robotwin_lora_train_cfg.cfg_prob = 0.1

# LoRA
va_robotwin_lora_train_cfg.use_lora = True
va_robotwin_lora_train_cfg.lora_rank = 32
va_robotwin_lora_train_cfg.lora_alpha = 32
va_robotwin_lora_train_cfg.lora_dropout = 0.0
va_robotwin_lora_train_cfg.full_train_action_io = True
va_robotwin_lora_train_cfg.full_train_action_cond_embedder = False
# Nothing updates the frozen base, so fp32 master copies of it are wasted:
# bf16 halves the base footprint from 10.2GB to 5.1GB per rank at world_size=2.
va_robotwin_lora_train_cfg.base_param_dtype = torch.bfloat16

# Training parameters.
# LoRA wants ~10x the full-finetune LR (1e-5), no weight decay (decaying
# lora_B toward zero fights the adapter), and a longer warmup because the
# adapter starts from a zero-output state.
va_robotwin_lora_train_cfg.learning_rate = 1e-4
va_robotwin_lora_train_cfg.beta1 = 0.9
va_robotwin_lora_train_cfg.beta2 = 0.95
va_robotwin_lora_train_cfg.weight_decay = 0.0
va_robotwin_lora_train_cfg.warmup_steps = 100
va_robotwin_lora_train_cfg.batch_size = 1
va_robotwin_lora_train_cfg.gradient_accumulation_steps = 32
va_robotwin_lora_train_cfg.num_steps = 50000

# Activation memory is the one cost LoRA does not reduce, and packed sequence
# length is unbounded in the dataset (measured RoboTwin range: 1408 / 3072 /
# 15232 tokens for min / median / max). Cap it so one long segment cannot OOM
# the run hours in. Raise once you have measured headroom on the target GPU.
va_robotwin_lora_train_cfg.max_seq_len = 8192
