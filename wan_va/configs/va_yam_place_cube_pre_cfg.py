# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference config for the YAM bimanual arms on the `place_cube` task -- FIRST recording.

This is the config for `place_cube_pre`: the original 57-episode place_cube recording
(Datasets/yam-real-world-lingbot-va/place_cube_pre), kept so the checkpoints trained on
it stay servable. `va_yam_place_cube_cfg` now describes the re-recorded 55-episode
`place_cube` dataset, whose norm_stat differs -- the two are not interchangeable.

`place_cube_pre` episodes are 259-651 frames (8.6-21.7s), so their latents are
extracted at stride 4 and a whole episode still fits comfortably in one action_config
segment. Stride is not a free knob -- it fixes `action_per_frame == 4 * stride`, and
that field is a single global the server uses to shape the actions it GENERATES, so
one config serves exactly one stride.

The 16-dim bimanual EEF layout, `used_action_channel_ids`, `env_type`
('robotwin_tshape'), `obs_cam_keys` and the camera geometry are the same as
`va_yam_place_cube_cfg`'s, since both recordings were made on the same arms and
converted by the same FK path (script/convert_yam_to_lingbot_va.py).

Every setting is stated here; only the universal defaults in shared_config.py come
from elsewhere. `va_yam_place_cube_pre_train_cfg` inherits from this config, so
training and serving cannot drift apart.
"""
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_yam_place_cube_pre_cfg = EasyDict(__name__='Config: VA yam place_cube_pre')
va_yam_place_cube_pre_cfg.update(va_shared_cfg)

# Built by script/assemble_yam_serving_model.sh from a place_cube_pre training
# checkpoint (under train_out/yam_place_cube_pre/); pass both arguments explicitly,
# since the script's defaults are the current place_cube's.
va_yam_place_cube_pre_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-yam-place-cube-pre')
va_yam_place_cube_pre_cfg.infer_mode = 'server'

# --- Embodiment and action layout ---
va_yam_place_cube_pre_cfg.env_type = 'robotwin_tshape'
va_yam_place_cube_pre_cfg.height = 256
va_yam_place_cube_pre_cfg.width = 320
va_yam_place_cube_pre_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_yam_place_cube_pre_cfg.action_dim = 30
va_yam_place_cube_pre_cfg.action_space = 'eef'
# [left EEF pose 0:7, left gripper 28, right EEF pose 7:14, right gripper 29]
va_yam_place_cube_pre_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_yam_place_cube_pre_cfg.used_action_channel_ids)
] * va_yam_place_cube_pre_cfg.action_dim
for i, j in enumerate(va_yam_place_cube_pre_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_yam_place_cube_pre_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_yam_place_cube_pre_cfg.action_norm_method = 'quantiles'
# Recomputed q01/q99 from place_cube_pre itself (script/convert_yam_to_lingbot_va.py), on
# the relative-pose-transformed action -- matching how _action_post_process applies
# get_relative_pose before normalization for env_type == 'robotwin_tshape'. These are
# NOT interchangeable with another task's or recording's: serving a checkpoint against
# the wrong quantiles denormalizes every action wrongly without raising.
#
# Fitted on EPISODES 0-49 ONLY (--norm-stat-episodes 0-49), matching
# va_yam_place_cube_pre_train_cfg.episode_subset. Episodes 50-56 are held out, and
# fitting the quantiles over them too would leak the held-out action range into the
# clip bounds. Refitting over all 57 moves the quantiles by up to 6.9% of span (left
# quat w), so the two are not interchangeable either: if you change the training
# split, re-run the converter with a matching --norm-stat-episodes and update both
# this field and the dataset's norm_stat.json, which records its own fit set.
va_yam_place_cube_pre_cfg.norm_stat = {
    "q01": [
        -0.00016070783138275147, -0.32244915748015046, -0.22268557082861662,
        -0.18129835732268507, -2.6905704188395483e-06, -0.13079127269402577,
        0.8282192791907785, -0.013634251356124878, -0.01888587432913482,
        -0.24117010924965143, -0.3471441539571246, -0.0046822951151956,
        -0.3121011685617784, 0.8140303019675041
    ] + [0.] * 14 + [0.19610601961612703, 0.2194398146867752],
    "q99": [
        0.30276643395423875, 0.020465014409273858, 0.0, 0.3317893134974764,
        0.4875007719758063, 0.3150782522744201, 1.0, 0.2645904517173767,
        0.3183043552190065, 0.002353653311729431, 0.15644031522200794,
        0.5667228822832248, 0.14347288023456786, 1.0
    ] + [0.] * 14 + [1.0, 0.9890429973602295],
}

# Invariant: action_per_frame == round(ori_fps / target_fps) * 4. Latents are
# extracted at stride 4 on a 30fps source (7.5fps), so 16. See va_yam_place_cube_cfg
# for why a stale value desyncs the closed loop.
va_yam_place_cube_pre_cfg.action_per_frame = 16
va_yam_place_cube_pre_cfg.frame_chunk_size = 2
va_yam_place_cube_pre_cfg.attn_window = 72

# --- Sampling ---
va_yam_place_cube_pre_cfg.guidance_scale = 5
va_yam_place_cube_pre_cfg.action_guidance_scale = 1
va_yam_place_cube_pre_cfg.num_inference_steps = 5
va_yam_place_cube_pre_cfg.video_exec_step = -1  # -1 means "run the full video schedule."
va_yam_place_cube_pre_cfg.action_num_inference_steps = 10
va_yam_place_cube_pre_cfg.snr_shift = 5.0
va_yam_place_cube_pre_cfg.action_snr_shift = 1.0
