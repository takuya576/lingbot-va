# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference config for YAM `place_cube` checkpoints trained on JOINT-space actions.

Identical to `va_yam_place_cube_base_cfg` -- same episodes, videos, cached latents,
stride, camera setup and inference settings -- except for what the actions are. The
EEF configs emit a 16-dim segment-relative end-effector pose that the client turns
into joints by IK; this one emits absolute YAM joints the client commands directly
(`yam_bridge/action_convert.YamJointActionConverter`), so there is no IK, no anchor
and no branch switching on the deploy side.

The actions sit in the base model's joint slots of the canonical 30-dim layout
(README "Action Format"; lingbot-va-base's action_embedder/action_proj_out carry
trained weights there, at the same norms as the EEF channels), not in the EEF slots
the other YAM configs use. YAM has 6 arm joints, so channels 20 and 27 -- each arm's
7th joint -- are unused, as the base model's pretraining padded every <7-DoF arm.

`used_action_channel_ids` is ordered [left q6, left grip, right q6, right grip],
which is both the dataset's raw 14-dim column layout and raiden's
`--action-type joint` command layout, so postprocess_action's output is a command.

Every setting is stated here; only the universal defaults in shared_config.py come
from elsewhere. `va_yam_place_cube_joint_train_cfg` inherits from this config, so
training and serving cannot drift apart.

Dataset: Datasets/yam-real-world-lingbot-va-joint/place_cube, built by
`script/convert_yam_to_lingbot_va.py --action-space joint --latents-from <EEF dataset>`.
It lives outside yam-real-world-lingbot-va/ on purpose: a dataset_path pointed at that
parent sweeps in every dataset beneath it (construct_lerobot_multi_processor).
"""
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_yam_place_cube_joint_cfg = EasyDict(__name__='Config: VA yam place_cube (joint)')
va_yam_place_cube_joint_cfg.update(va_shared_cfg)

va_yam_place_cube_joint_cfg.wan22_pretrained_model_name_or_path = (
    'Pretrained_models/lingbot-va-posttrain-yam-place-cube-joint')
va_yam_place_cube_joint_cfg.infer_mode = 'server'

# --- Embodiment and action layout ---
# env_type stays 'robotwin_tshape' although nothing is relativized in joint space: it
# also fixes the camera canvas the cached latents were extracted with.
va_yam_place_cube_joint_cfg.env_type = 'robotwin_tshape'
va_yam_place_cube_joint_cfg.height = 256
va_yam_place_cube_joint_cfg.width = 320
va_yam_place_cube_joint_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_yam_place_cube_joint_cfg.action_dim = 30
va_yam_place_cube_joint_cfg.action_space = 'joint'
# [left joints 14:20, left gripper 28, right joints 21:27, right gripper 29]
va_yam_place_cube_joint_cfg.used_action_channel_ids = list(range(14, 20)) + list(
    range(28, 29)) + list(range(21, 27)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_yam_place_cube_joint_cfg.used_action_channel_ids)
] * va_yam_place_cube_joint_cfg.action_dim
for i, j in enumerate(va_yam_place_cube_joint_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_yam_place_cube_joint_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_yam_place_cube_joint_cfg.action_norm_method = 'quantiles'
# q01/q99 of ABSOLUTE joint angles (rad) and grippers, from
# `convert_yam_to_lingbot_va.py --action-space joint --norm-stat-episodes 0-49`: the
# same fit set as the EEF configs' episode_subset, so episodes 50-54 stay held out.
# The grippers are bit-identical to va_yam_place_cube_cfg's, as they should be --
# neither conversion transforms them. Matches the dataset's norm_stat.json, which
# eval_yam_openloop.py checks before scoring.
va_yam_place_cube_joint_cfg.norm_stat = {
    "q01": [0.] * 14 + [
        -0.6418325901031494, 0.004439123906195164, 0.00667582219466567,
        -0.8634698987007141, -0.30221322178840637, -0.4910217523574829, 0.0,
        -0.04901960864663124, -0.005149920005351305, 0.00667582219466567,
        -0.81170654296875, -0.575509250164032, -0.5270618796348572, 0.0,
        0.1368102729320526, 0.09933413565158844
    ],
    "q99": [0.] * 14 + [
        0.16846565902233124, 2.4749298095703125, 1.8889834880828857,
        0.5079346895217896, 0.6308234333992004, 0.5884796977043152, 0.0,
        0.6288624405860901, 2.494659423828125, 1.7729076147079468,
        0.5121309161186218, 0.43062537908554077, 0.5978057384490967, 0.0,
        1.0, 0.9934257864952087
    ],
}

# Invariant: action_per_frame == round(ori_fps / target_fps) * 4. The joint dataset
# reuses the EEF conversion's stride 4 latents on a 30fps source, so 16.
va_yam_place_cube_joint_cfg.action_per_frame = 16
va_yam_place_cube_joint_cfg.frame_chunk_size = 2
va_yam_place_cube_joint_cfg.attn_window = 72

# --- Sampling ---
va_yam_place_cube_joint_cfg.guidance_scale = 5
va_yam_place_cube_joint_cfg.action_guidance_scale = 1
va_yam_place_cube_joint_cfg.num_inference_steps = 5
va_yam_place_cube_joint_cfg.video_exec_step = -1  # -1 means "run the full video schedule."
va_yam_place_cube_joint_cfg.action_num_inference_steps = 10
va_yam_place_cube_joint_cfg.snr_shift = 5.0
va_yam_place_cube_joint_cfg.action_snr_shift = 1.0
