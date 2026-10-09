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

# Inference: how the first chunk treats latent frame 0's action slot, which training
# fills with a pad (raw zeros, normalized and clipped).
#   'denoised'  denoised along with the rest of the chunk, as training always has it --
#               training never shows the model a clean token inside the chunk it is
#               denoising.
#   'train_pad' held to the pad at timestep 0. Leaves a ~10 mm first-step bias on
#               YAM place_cube (vs ~2-3 mm denoised), mostly backward, which drove
#               the idle arm into its fold stop.
# Either way, frame 0 of the history fed back afterwards is the exact pad.
# Override at launch with wan_va_server --frame0-action.
va_shared_cfg.frame0_action = 'denoised'

# --- LoRA parameter-efficient post-training (off by default) ---
# `base_param_dtype` is the dtype the transformer is loaded in for training.
# float32 reproduces the original full-finetune numerics; bfloat16 halves the
# frozen-base footprint and is what LoRA runs want, since nothing updates the
# base weights so fp32 master copies of them are pure waste.
va_shared_cfg.base_param_dtype = torch.float32
va_shared_cfg.use_lora = False
va_shared_cfg.lora_rank = 32
va_shared_cfg.lora_alpha = 64
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

# What the model's actions mean, for the YAM embodiment. 'eef' is the end-effector
# pose layout: for env_type == 'robotwin_tshape', _action_post_process relativizes it
# against each segment's first frame, and the YAM client converts it to joints via IK
# (yam_bridge/action_convert.YamActionConverter). 'joint' is absolute joint angles in
# the base model's joint slots (14:20 left, 21:27 right): never relativized, and the
# client commands them directly (YamJointActionConverter). env_type stays
# 'robotwin_tshape' in both, since it also fixes the camera canvas the cached latents
# were extracted with. Reported to clients in the handshake (served_metadata).
va_shared_cfg.action_space = 'eef'

# Skip training samples whose packed sequence exceeds this many tokens, to bound
# activation memory. None disables the guard.
va_shared_cfg.max_seq_len = None

# Train on only these episodes; None (the default) means all of them. This is the
# repo's only train/val split mechanism -- there is no validation loop, and the
# `splits` field lerobot writes into meta/info.json is read by nothing.
#
# Values are ABSOLUTE `episode_index` values as they appear in meta/episodes.jsonl,
# not positions, and they need not be contiguous. Held-out episodes keep their
# parquet rows, videos and cached latents on disk; they are simply never sampled,
# so an offline evaluator (script/eval_yam_openloop.py) can still read them.
#
# The filter applies to EVERY repo discovered under dataset_path, which is only
# unambiguous because a per-task config points dataset_path at a single task
# directory. Do not set this on a config whose dataset_path spans several tasks.
va_shared_cfg.episode_subset = None
