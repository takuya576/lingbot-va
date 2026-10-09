# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference config for the YAM bimanual arms on the `place_cube` task.

This is the config the policy server must be launched with to serve a place_cube
checkpoint (`CONFIG_NAME=yam_place_cube`). Serving a YAM checkpoint under the
`robotwin` config would load RoboTwin's q01/q99 and silently denormalize every action
against the wrong quantiles -- the failure mode is plausible-looking motion, not an
error.

This is the re-recorded 55-episode `place_cube` dataset. The original 57-episode
recording and the checkpoints trained on it are served by `va_yam_place_cube_pre_cfg`.

`place_cube` episodes are 193-657 frames (6.4-21.9s), so their latents are extracted
at stride 4 and a whole episode still fits comfortably in one action_config segment.
Stride is not a free knob -- it fixes `action_per_frame == 4 * stride`, and that field
is a single global the server uses to shape the actions it GENERATES, so one config
serves exactly one stride. script/extract_va_latents.py must therefore be run with
`--stride 4`: its default is 5fps, i.e. stride 6 on this 30fps source, and nothing
downstream checks the stride of the cached latents against action_per_frame.

YAM joint angles are FK-converted to the same 16-dim bimanual EEF layout as robotwin
(see script/convert_yam_to_lingbot_va.py), so `used_action_channel_ids`, `env_type`
('robotwin_tshape') and `obs_cam_keys` take va_robotwin_cfg's values. The model emits
EEF pose, not joints. Converting its output into YAM joint commands (un-relativize
against the rollout anchor, then IK on `grasp_site`) is `yam_bridge/action_convert.py`.

Every setting is stated here; only the universal defaults in shared_config.py come
from elsewhere. `va_yam_place_cube_train_cfg` inherits from this config, so training
and serving cannot drift apart.
"""
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_yam_place_cube_cfg = EasyDict(__name__='Config: VA yam place_cube')
va_yam_place_cube_cfg.update(va_shared_cfg)

# Built by script/assemble_yam_serving_model.sh from a place_cube training checkpoint.
va_yam_place_cube_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-yam-place-cube')
va_yam_place_cube_cfg.infer_mode = 'server'

# --- Embodiment and action layout ---
va_yam_place_cube_cfg.env_type = 'robotwin_tshape'
va_yam_place_cube_cfg.height = 256
va_yam_place_cube_cfg.width = 320
va_yam_place_cube_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_yam_place_cube_cfg.action_dim = 30
va_yam_place_cube_cfg.action_space = 'eef'
# [left EEF pose 0:7, left gripper 28, right EEF pose 7:14, right gripper 29]
va_yam_place_cube_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_yam_place_cube_cfg.used_action_channel_ids)
] * va_yam_place_cube_cfg.action_dim
for i, j in enumerate(va_yam_place_cube_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_yam_place_cube_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_yam_place_cube_cfg.action_norm_method = 'quantiles'
# Recomputed q01/q99 from place_cube itself (script/convert_yam_to_lingbot_va.py), on
# the relative-pose-transformed action -- matching how _action_post_process applies
# get_relative_pose before normalization for env_type == 'robotwin_tshape'. These are
# NOT interchangeable with another task's or recording's: serving a checkpoint against
# the wrong quantiles denormalizes every action wrongly without raising.
#
# Fitted on EPISODES 0-49 ONLY (--norm-stat-episodes 0-49), matching
# va_yam_place_cube_train_cfg.episode_subset. Episodes 50-54 are held out, and
# fitting the quantiles over them too would leak the held-out action range into the
# clip bounds. Refitting over all 55 moves the quantiles by up to 3.2% of span, so
# the two are not interchangeable either: if you change the training split, re-run
# the converter with a matching --norm-stat-episodes and update both this field and
# the dataset's norm_stat.json, which records its own fit set.
va_yam_place_cube_cfg.norm_stat = {
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

# Actions are NOT downsampled with the video: _action_post_process keeps them at the
# raw 30fps and packs frame_stride*4 of them per latent frame, so 16 at stride 4. The
# dataloader takes this from the data, but wan_va_server.py shapes the actions it
# GENERATES from this field, so a stale value would desync the closed loop: the world
# model would advance by one latent frame while the robot executed a different span
# of time before the next observation.
# Invariant: action_per_frame == round(ori_fps / target_fps) * 4.
va_yam_place_cube_cfg.action_per_frame = 16
va_yam_place_cube_cfg.frame_chunk_size = 2
va_yam_place_cube_cfg.attn_window = 72

# --- Sampling ---
va_yam_place_cube_cfg.guidance_scale = 5
va_yam_place_cube_cfg.action_guidance_scale = 1
va_yam_place_cube_cfg.num_inference_steps = 5
va_yam_place_cube_cfg.video_exec_step = -1  # -1 means "run the full video schedule."
va_yam_place_cube_cfg.action_num_inference_steps = 10
va_yam_place_cube_cfg.snr_shift = 5.0
va_yam_place_cube_cfg.action_snr_shift = 1.0
