# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = False

# --- LoRA parameter-efficient post-training (off by default) ---
# `base_param_dtype` is the dtype the transformer is loaded in for training.
# float32 reproduces the original full-finetune numerics; bfloat16 halves the
# frozen-base footprint and is what LoRA runs want, since nothing updates the
# base weights so fp32 master copies of them are pure waste.
va_shared_cfg.base_param_dtype = torch.float32
va_shared_cfg.use_lora = False
va_shared_cfg.lora_rank = 32
va_shared_cfg.lora_alpha = 32
va_shared_cfg.lora_dropout = 0.0
# Modules that get a low-rank adapter. Matched as suffixes of the module path.
va_shared_cfg.lora_target_modules = [
    'attn1.to_q', 'attn1.to_k', 'attn1.to_v', 'attn1.to_out.0',
    'attn2.to_q', 'attn2.to_k', 'attn2.to_v', 'attn2.to_out.0',
    'ffn.net.0.proj', 'ffn.net.2'
]
# Modules trained FULLY, with no adapter, while the rest of the base is frozen.
#
# action_embedder (30->3072) and action_proj_out (3072->30) total 0.19M params.
# A low-rank factorisation of a matrix whose narrow side is 30 would be higher
# rank than the matrix itself, so an adapter there is pointless -- and training
# them outright costs only ~9MB including optimizer state. Hence: no LoRA, full
# training. Needed whenever the action channel layout or normalisation differs
# from the checkpoint's.
va_shared_cfg.full_train_action_io = True
# condition_embedder_action is 88.9M; training it costs ~1.1GB/GPU of grads and
# Adam state. Off by default: the action noise schedule is unchanged, so there
# is nothing new for it to learn.
va_shared_cfg.full_train_action_cond_embedder = False

# Skip training samples whose packed sequence exceeds this many tokens, to bound
# activation memory. None disables the guard.
va_shared_cfg.max_seq_len = None