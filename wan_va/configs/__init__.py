# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_yam_place_cube_cfg import va_yam_place_cube_cfg
from .va_yam_place_cube_train_cfg import va_yam_place_cube_train_cfg
from .va_yam_place_cube_pre_cfg import va_yam_place_cube_pre_cfg
from .va_yam_place_cube_pre_train_cfg import va_yam_place_cube_pre_train_cfg
from .va_yam_place_cube_base_cfg import va_yam_place_cube_base_cfg
from .va_yam_place_cube_base_train_cfg import va_yam_place_cube_base_train_cfg
from .va_yam_place_cube_joint_cfg import va_yam_place_cube_joint_cfg
from .va_yam_place_cube_joint_train_cfg import va_yam_place_cube_joint_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'yam_place_cube': va_yam_place_cube_cfg,
    'yam_place_cube_train': va_yam_place_cube_train_cfg,
    'yam_place_cube_pre': va_yam_place_cube_pre_cfg,
    'yam_place_cube_pre_train': va_yam_place_cube_pre_train_cfg,
    'yam_place_cube_base': va_yam_place_cube_base_cfg,
    'yam_place_cube_base_train': va_yam_place_cube_base_train_cfg,
    'yam_place_cube_joint': va_yam_place_cube_joint_cfg,
    'yam_place_cube_joint_train': va_yam_place_cube_joint_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
}
