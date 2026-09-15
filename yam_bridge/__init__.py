# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Bridge between YAM robot joint space and the LingBot-VA 16-dim EEF action space.

Deliberately **torch-free and outside `wan_va/`**. Both directions of the conversion
are needed in environments that have no deep-learning stack:

  - `script/convert_yam_to_lingbot_va.py` prepares training data with FK, and is
    documented to run in a lightweight env (numpy/scipy/mujoco/mink, no torch).
  - The offline validation and the deployment-side IK run against the YAM robot env
    (raiden's `.venv`), which has mujoco/mink but not diffusers or easydict.

`wan_va/__init__.py` eagerly imports `configs`, `distributed` and `modules`, so any
module placed under `wan_va/` drags in the full training stack on import. Keeping this
package separate is what lets one shared kinematics implementation serve both sides --
which is the whole point, since a mismatch between train-time FK and deploy-time IK
would show up as a silent constant offset.
"""
