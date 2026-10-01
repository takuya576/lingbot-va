# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Post-training config for YAM `place_cube`, initialized from `lingbot-va-base`.

`va_yam_place_cube_train_cfg` starts from `lingbot-va-posttrain-robotwin`, which has
already been post-trained on RoboTwin. This one starts from the pre-trained
`lingbot-va-base` (robbyant/lingbot-va-base, rev 68b7bc1b) instead. EVERYTHING else --
dataset, episode split, norm_stat, LoRA shape, optimizer, steps, max_seq_len -- is
inherited unchanged, so the initialization is the only variable between the two runs.

The cached place_cube latents, text embeddings and empty_emb.pt are reused as-is. That
is only valid because the two checkpoints differ in `transformer/` alone: every file of
`vae/`, `text_encoder/` and `tokenizer/` has the same sha256 on both HF repos, and
Pretrained_models/lingbot-va-base hard-links them from the robotwin copy. A base whose
VAE or T5 differed would need latents and embeddings re-extracted.
"""
from easydict import EasyDict

from .va_yam_place_cube_train_cfg import va_yam_place_cube_train_cfg

va_yam_place_cube_base_train_cfg = EasyDict(
    __name__='Config: VA yam place_cube train (base init)')
va_yam_place_cube_base_train_cfg.update(va_yam_place_cube_train_cfg)

# The un-adapted base, not the serving directory of va_yam_place_cube_base_cfg (that is
# this run's output). Its transformer/config.json says attn_mode "flex" where robotwin's
# says "torch"; neither is read, since train.py and wan_va_server.py both pass
# attn_mode explicitly to from_pretrained.
va_yam_place_cube_base_train_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-base')

# Separate from train_out/yam_place_cube, so that assemble_yam_serving_model.sh's
# "newest checkpoint" default can never pick a base-initialized run for the
# robotwin-initialized serving directory, or the reverse.
va_yam_place_cube_base_train_cfg.save_root = './train_out/yam_place_cube_base'
