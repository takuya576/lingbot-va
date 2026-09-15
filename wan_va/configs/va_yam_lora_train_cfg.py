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
from .va_yam_cfg import va_yam_cfg
import os
import torch

va_yam_lora_train_cfg = EasyDict(__name__='Config: VA yam LoRA train')
va_yam_lora_train_cfg.update(va_yam_cfg)

# action_per_frame, norm_stat, the EEF channel layout and the camera setup all come
# from va_yam_cfg, so training and serving cannot drift apart. Only the dataset and
# the optimization settings are training-specific.

# va_yam_cfg points this at the LoRA-merged serving directory, which is the *output*
# of this run. Training must start from the original un-adapted base instead -- LoRA
# injection asserts it (wan_va/modules/lora.py:223), and resuming against merged
# weights would fold the adapter in a second time.
va_yam_lora_train_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-robotwin')

va_yam_lora_train_cfg.dataset_path = '/home/takuya/lingbot-va/Datasets/yam-real-world-lingbot-va'
va_yam_lora_train_cfg.empty_emb_path = os.path.join(
    va_yam_lora_train_cfg.dataset_path, 'cover_blocks', 'empty_emb.pt')
va_yam_lora_train_cfg.save_root = './train_out/yam_lora'
va_yam_lora_train_cfg.enable_wandb = True
va_yam_lora_train_cfg.load_worker = 16
va_yam_lora_train_cfg.save_interval = 1000
va_yam_lora_train_cfg.gc_interval = 50
va_yam_lora_train_cfg.cfg_prob = 0.1

# LoRA
va_yam_lora_train_cfg.use_lora = True
va_yam_lora_train_cfg.lora_rank = 32
va_yam_lora_train_cfg.lora_alpha = 64
va_yam_lora_train_cfg.lora_dropout = 0.0
va_yam_lora_train_cfg.full_train_action_io = True
va_yam_lora_train_cfg.full_train_action_cond_embedder = False
# Nothing updates the frozen base, so fp32 master copies of it are wasted:
# bf16 halves the base footprint from 10.2GB to 5.1GB per rank at world_size=2.
va_yam_lora_train_cfg.base_param_dtype = torch.bfloat16

# Training parameters.
# LoRA wants ~10x the full-finetune LR (1e-5), no weight decay (decaying
# lora_B toward zero fights the adapter), and a longer warmup because the
# adapter starts from a zero-output state.
va_yam_lora_train_cfg.learning_rate = 1e-4
va_yam_lora_train_cfg.beta1 = 0.9
va_yam_lora_train_cfg.beta2 = 0.95
va_yam_lora_train_cfg.weight_decay = 0.0
va_yam_lora_train_cfg.warmup_steps = 20
va_yam_lora_train_cfg.batch_size = 1
va_yam_lora_train_cfg.gradient_accumulation_steps = 16
va_yam_lora_train_cfg.num_steps = 1000

# Activation memory is the one cost LoRA does not reduce, and packed sequence
# length is unbounded in the dataset. cover_blocks episodes are full 40-70s
# multi-step tasks (unlike robotwin's ~3s atomic-action episodes), so at
# robotwin's 12-15fps sampling one episode would need 37k-70k tokens. Latents are
# instead extracted at 5fps (stride 6, see script/extract_va_latents.py), which
# fits a WHOLE episode in one action_config segment: 14112 tokens for the
# shortest episode, 16704 median, 26496 for the longest. Sized to admit all of
# them, because a whole-episode segment is anchored at frame 0 -- the same anchor
# the deployment client holds for an entire rollout (it captures init_eef_pose
# once and never re-anchors). Chunking episodes instead would train half the
# samples against mid-episode anchors.
va_yam_lora_train_cfg.max_seq_len = 26624
