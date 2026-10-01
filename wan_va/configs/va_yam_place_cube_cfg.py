# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference config for the YAM bimanual arms on the `place_cube` task.

This is the re-recorded 55-episode `place_cube` dataset. The original 57-episode
recording and the checkpoints trained on it are served by `va_yam_place_cube_pre_cfg`.

Separate from `va_yam_cover_blocks_cfg` (which is `cover_blocks`) for one reason: latent frame
stride. `place_cube` episodes are 193-657 frames (6.4-21.9s), a third of the length
of `cover_blocks`', so their latents are extracted at stride 4 rather than stride 6
and a whole episode still fits comfortably in one action_config segment. Stride is
not a free knob -- it fixes `action_per_frame == 4 * stride`, and that field is a
single global the server uses to shape the actions it GENERATES. One config cannot
serve both strides, so the two tasks get one config each.

script/extract_va_latents.py must therefore be run with `--stride 4`: its default is
5fps, i.e. stride 6 on this 30fps source, and nothing downstream checks the stride of
the cached latents against action_per_frame.

Everything else -- the 16-dim bimanual EEF layout, `used_action_channel_ids`,
`env_type` ('robotwin_tshape'), `obs_cam_keys`, the camera geometry -- is inherited
from `va_yam_cover_blocks_cfg` unchanged, since both tasks were recorded on the same arms and
converted by the same FK path (script/convert_yam_to_lingbot_va.py).
"""
from easydict import EasyDict
from .va_yam_cover_blocks_cfg import va_yam_cover_blocks_cfg

va_yam_place_cube_cfg = EasyDict(__name__='Config: VA yam place_cube')
va_yam_place_cube_cfg.update(va_yam_cover_blocks_cfg)

# Built by script/assemble_yam_serving_model.sh from a place_cube training checkpoint.
va_yam_place_cube_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-yam-place-cube')

# Invariant: action_per_frame == round(ori_fps / target_fps) * 4. Latents are
# extracted at stride 4 on a 30fps source (7.5fps), so 16 -- not va_yam_cover_blocks_cfg's 24.
# Stated explicitly rather than left to va_robotwin_cfg's identical default, because
# the value is only correct here by way of the stride, not by inheritance.
va_yam_place_cube_cfg.action_per_frame = 16

va_yam_place_cube_cfg.num_inference_steps = 5
va_yam_place_cube_cfg.video_exec_step = -1  # -1 means "run the full video schedule."
va_yam_place_cube_cfg.action_num_inference_steps = 10

# Recomputed q01/q99 from place_cube itself (script/convert_yam_to_lingbot_va.py), on
# the relative-pose-transformed action -- matching how _action_post_process applies
# get_relative_pose before normalization for env_type == 'robotwin_tshape'. These are
# NOT interchangeable with va_yam_cover_blocks_cfg's: serving a place_cube checkpoint against
# cover_blocks' quantiles denormalizes every action wrongly without raising.
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
