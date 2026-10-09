# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference config for YAM `place_cube` checkpoints post-trained from `lingbot-va-base`.

Identical to `va_yam_place_cube_cfg` -- same dataset, so the same norm_stat,
action_per_frame and camera setup -- except for the serving directory. The two exist
separately only so that a base-initialized checkpoint and a robotwin-initialized one
(`va_yam_place_cube_cfg`) can both stay assembled and be served side by side. What
initialized the transformer is invisible once LoRA is merged, so nothing but this path
tells the two apart at serving time. Keep the two in step when changing either.

Every setting is stated here; only the universal defaults in shared_config.py come
from elsewhere. `va_yam_place_cube_base_train_cfg` inherits from this config, so
training and serving cannot drift apart.
"""
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_yam_place_cube_base_cfg = EasyDict(__name__='Config: VA yam place_cube (base init)')
va_yam_place_cube_base_cfg.update(va_shared_cfg)

# Built by script/assemble_yam_serving_model.sh from a checkpoint under
# train_out/yam_place_cube_base/; pass both arguments explicitly, since the script's
# defaults are the robotwin-initialized place_cube's.
va_yam_place_cube_base_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-yam-place-cube-base')
va_yam_place_cube_base_cfg.infer_mode = 'server'

# --- Embodiment and action layout ---
va_yam_place_cube_base_cfg.env_type = 'robotwin_tshape'
va_yam_place_cube_base_cfg.height = 256
va_yam_place_cube_base_cfg.width = 320
va_yam_place_cube_base_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_yam_place_cube_base_cfg.action_dim = 30
va_yam_place_cube_base_cfg.action_space = 'eef'
# [left EEF pose 0:7, left gripper 28, right EEF pose 7:14, right gripper 29]
va_yam_place_cube_base_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_yam_place_cube_base_cfg.used_action_channel_ids)
] * va_yam_place_cube_base_cfg.action_dim
for i, j in enumerate(va_yam_place_cube_base_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_yam_place_cube_base_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_yam_place_cube_base_cfg.action_norm_method = 'quantiles'
# The same place_cube quantiles as va_yam_place_cube_cfg (fitted on episodes 0-49 of
# the same dataset); see there for provenance.
va_yam_place_cube_base_cfg.norm_stat = {
    "q01": [
        -0.01613114058971405, -0.34552717083133755, -0.19018375987187028,
        -0.11398230924629182, -4.057140035762635e-05, -0.0917769620582623,
        0.7891655174002978, -0.020370704233646394, -0.032947961343452334,
        -0.1863137216307223, -0.4700328600521805, -0.00012245928614314176,
        -0.187849806783216, 0.7876441538305189
    ] + [0.] * 14 + [0.13681027710437774, 0.09933413565158844],
    "q99": [
        0.38210201799869514, 0.03854033527430142, 0.09643310487270347,
        0.4662490917757731, 0.5650704936544517, 0.21289535581749536, 1.0,
        0.37775842815637584, 0.3253298039082438, 0.08556239753961561,
        0.23916610878538516, 0.566232960392461, 0.14781434245087266, 1.0
    ] + [0.] * 14 + [1.0, 0.9934257864952087],
}

# Invariant: action_per_frame == round(ori_fps / target_fps) * 4. Stride 4 latents on
# a 30fps source, so 16.
va_yam_place_cube_base_cfg.action_per_frame = 16
va_yam_place_cube_base_cfg.frame_chunk_size = 2
va_yam_place_cube_base_cfg.attn_window = 72

# --- Sampling ---
va_yam_place_cube_base_cfg.guidance_scale = 5
va_yam_place_cube_base_cfg.action_guidance_scale = 1
va_yam_place_cube_base_cfg.num_inference_steps = 5
va_yam_place_cube_base_cfg.video_exec_step = -1  # -1 means "run the full video schedule."
va_yam_place_cube_base_cfg.action_num_inference_steps = 10
va_yam_place_cube_base_cfg.snr_shift = 5.0
va_yam_place_cube_base_cfg.action_snr_shift = 1.0
