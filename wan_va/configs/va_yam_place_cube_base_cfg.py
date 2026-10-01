# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference config for YAM `place_cube` checkpoints post-trained from `lingbot-va-base`.

Identical to `va_yam_place_cube_cfg` -- same dataset, so the same norm_stat,
action_per_frame and camera setup -- except for the serving directory. The two exist
separately only so that a base-initialized checkpoint and a robotwin-initialized one
(`va_yam_place_cube_cfg`) can both stay assembled and be served side by side. What
initialized the transformer is invisible once LoRA is merged, so nothing but this path
tells the two apart at serving time.
"""
from easydict import EasyDict
from .va_yam_place_cube_cfg import va_yam_place_cube_cfg

va_yam_place_cube_base_cfg = EasyDict(__name__='Config: VA yam place_cube (base init)')
va_yam_place_cube_base_cfg.update(va_yam_place_cube_cfg)

# Built by script/assemble_yam_serving_model.sh from a checkpoint under
# train_out/yam_place_cube_base/; pass both arguments explicitly, since the script's
# defaults are the robotwin-initialized place_cube's.
va_yam_place_cube_base_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-yam-place-cube-base')
