# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference config for the YAM bimanual arms, `cover_blocks` task.

This is the config the policy server must be launched with to serve a cover_blocks
checkpoint (`CONFIG_NAME=yam_cover_blocks`). Serving a YAM checkpoint under the
`robotwin` config would load RoboTwin's q01/q99 and silently denormalize every action
against the wrong quantiles -- the failure mode is plausible-looking motion, not an
error. The same goes for serving it under `yam_place_cube`, whose norm_stat is fitted
on a different task.

It carries the embodiment-level facts shared with training, so
`va_yam_cover_blocks_train_cfg` inherits from here rather than restating them: a
`norm_stat` that differs between training and serving is exactly the kind of mismatch
that never raises and never gets noticed.

cover_blocks joint angles are FK-converted to the same 16-dim bimanual EEF layout as
robotwin (see script/convert_yam_to_lingbot_va.py), so `used_action_channel_ids`,
`env_type` ('robotwin_tshape') and `obs_cam_keys` are inherited unchanged from
va_robotwin_cfg.

The model emits EEF pose, not joints. Converting its output into YAM joint commands
(un-relativize against the rollout anchor, then IK on `grasp_site`) is
`yam_bridge/action_convert.py`.
"""
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg

va_yam_cover_blocks_cfg = EasyDict(__name__='Config: VA yam cover_blocks')
va_yam_cover_blocks_cfg.update(va_robotwin_cfg)

# Built by script/assemble_yam_serving_model.sh: the LoRA-merged transformer from a
# training checkpoint, plus vae/tokenizer/text_encoder symlinked from the base model.
va_yam_cover_blocks_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-yam-cover-blocks')

# Actions are NOT downsampled with the video: _action_post_process keeps them at the
# raw 30fps and packs frame_stride*4 of them per latent frame. At 5fps extraction
# (stride 6) that is 24, not robotwin's 16 (stride 4). The dataloader takes this from
# the data, but wan_va_server.py shapes the actions it GENERATES from this field, so
# a stale 16 would desync the closed loop: the world model advances 0.8s per latent
# frame while the robot executes only 16/30 = 0.53s before the next observation.
# Invariant: action_per_frame == round(ori_fps / target_fps) * 4.
va_yam_cover_blocks_cfg.action_per_frame = 24

va_yam_cover_blocks_cfg.num_inference_steps = 5
va_yam_cover_blocks_cfg.video_exec_step = -1 # -1 means “run the full video schedule.”
va_yam_cover_blocks_cfg.action_num_inference_steps = 10

# Recomputed q01/q99 from cover_blocks itself (script/convert_yam_to_lingbot_va.py),
# on the relative-pose-transformed action (matches how _action_post_process applies
# get_relative_pose before normalization for env_type == 'robotwin_tshape').
va_yam_cover_blocks_cfg.norm_stat = {
    "q01": [
        -0.028332013487815853, -0.22303730249404907, -0.08279160648584366,
        -0.16091287598791865, -0.05368990912258327, -0.15764772348012823,
        0.6524848421224007, -0.02946704626083374, -0.037478614449501034,
        -0.1410510104894638, -0.27071777073164394, -0.11739074985787681,
        -0.10904350797740735, 0.6216650149760935
    ] + [0.] * 14 + [0.0, 0.004911225140094757],
    "q99": [
        0.24112678110599517, 0.01414784628897906, 0.21138109982013703,
        0.17418862743048596, 0.7524814535827876, 0.10980641370945785, 1.0,
        0.24038922786712646, 0.3230984127521515, 0.2385672795772553,
        0.1438824009114649, 0.7778388360651676, 0.19367477896099056, 1.0
    ] + [0.] * 14 + [1.0, 0.9912344217300415],
}
