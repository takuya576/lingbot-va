# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Convert LingBot-VA action output into YAM joint commands.

The model does not emit joint angles. For `env_type == 'robotwin_tshape'` it emits a
16-dim bimanual **end-effector pose**, expressed **relative to the pose at rollout
start** (`wan_va/dataset/lerobot_latent_dataset.py:55-68` applies `get_relative_pose`
before normalization; `wan_va/wan_va_server.py:243-253` denormalizes but never undoes
the relativization). YAM, meanwhile, is commanded in absolute joint space.

So the deployment chain is::

    model 16-dim relative pose
      -> un-relativize against the anchor latched at rollout start
      -> renormalize quaternions
      -> per-arm 4x4 SE(3)
      -> IK on grasp_site, seeded from the previous command
      -> (14,) joint command: [left q6, left gripper, right q6, right gripper]

The 16-dim channel order comes from `used_action_channel_ids` in
`wan_va/configs/va_robotwin_cfg.py:33-40`, which scatters the canonical 30-dim vector
down to `[0..6, 28, 7..13, 29]`::

    [0:3]   left xyz          relative
    [3:7]   left quat xyzw    relative
    [7]     left gripper      absolute, [0, 1]
    [8:11]  right xyz         relative
    [11:15] right quat xyzw   relative
    [15]    right gripper     absolute, [0, 1]

Each arm's pose is in its own base frame -- both in the training data (the converter
runs a single-arm FK for both arms) and in raiden -- so no bimanual extrinsics are
involved anywhere in this file.

Checkpoints trained with `action_space = 'joint'` (wan_va/configs/shared_config.py)
skip all of the above: they emit absolute YAM joints, already in the 14-dim command
layout, and `YamJointActionConverter` only clips them to the hardware's limits.
`make_converter` picks between the two from what the policy server reports. Both
expose `predicted_rel16`, so plots and EEF error metrics read the same either way.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from yam_bridge.kinematics import ARM_DOF, YamKinematics

DEFAULT_I2RT_ROOT = "/data2/takuya/raiden/third_party/i2rt/i2rt"

# 16-dim EEF layout (the model's `used_action_channel_ids` output order).
LEFT_ARM = slice(0, 8)
RIGHT_ARM = slice(8, 16)
POSE = slice(0, 7)
GRIPPER = 7

# 14-dim YAM joint layout: left arm first, then right.
# `raiden/server.py:604`'s metadata string says "right_joints(7)+left_joints(7)", but
# every consumer in raiden treats [:7] as left (`_smooth_command` :908-923,
# `_check_joint_delta` :959-963). The code wins; this file is left-first.
YAM_DOF = 7
BIMANUAL_DOF = 14


def get_relative_pose(pose7: np.ndarray) -> np.ndarray:
    """Make an (N, 7) `[xyz, quat_xyzw]` sequence relative to its own first frame.

    Exact port of `wan_va/dataset/lerobot_latent_dataset.py:55-68`, which the training
    dataloader applies for `env_type == 'robotwin_tshape'` before normalization. Kept
    here beside `add_eef_pose`, the operation that inverts it, so the two conventions
    cannot drift apart -- `test_inverse_identity` in the validation script pins that.
    """
    pose7 = np.asarray(pose7, dtype=np.float64)
    rot = R.from_quat(pose7[:, 3:7])
    first_rot = R.from_quat(np.tile(pose7[:1, 3:7], (pose7.shape[0], 1)))
    relative_trans = pose7[:, :3] - pose7[0:1, :3]
    relative_quat = (first_rot.inv() * rot).as_quat()
    return np.concatenate([relative_trans, relative_quat], axis=1)


def relativize_action16(action16: np.ndarray) -> np.ndarray:
    """(N, 16) absolute bimanual EEF -> relative, grippers passed through untouched.

    Mirrors `_action_post_process` (`lerobot_latent_dataset.py:260-263`), which
    relativizes `[:, :7]` and `[:, 8:15]` and leaves the two gripper channels alone.
    """
    action16 = np.asarray(action16, dtype=np.float64)
    return np.concatenate([
        get_relative_pose(action16[:, 0:7]),
        action16[:, 7:8],
        get_relative_pose(action16[:, 8:15]),
        action16[:, 15:16],
    ], axis=1)


def add_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """Compose one arm's relative pose onto its anchor. Copied from RoboTwin's client.

    From `evaluation/robotwin/eval_polict_client_openpi.py:458-463`. Copied rather than
    imported because that module runs `os.chdir` into a hardcoded RoboTwin checkout at
    import time (lines 9-15), so importing it fails unless that checkout exists.

    Both arguments are 8-dim `[xyz(3), quat_xyzw(4), gripper(1)]`. Translation adds in
    the arm base frame and rotation composes as `anchor_R * relative_R`, which is the
    exact inverse of `get_relative_pose`'s `first_rot.inv() * rot`. The gripper is
    taken from `new_pose` untouched -- it was never relativized in the first place.
    """
    new_pose_R = R.from_quat(new_pose[3:7][None])
    init_pose_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def add_init_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    """Bimanual `add_eef_pose`: 16-dim relative -> 16-dim absolute."""
    left = add_eef_pose(new_pose[LEFT_ARM], init_pose[LEFT_ARM])
    right = add_eef_pose(new_pose[RIGHT_ARM], init_pose[RIGHT_ARM])
    return np.concatenate([left, right])


def pose_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """`[xyz, quat_xyzw]` -> 4x4, renormalizing the quaternion.

    The model's raw output is not unit-norm (it is a denormalized regression target),
    so normalizing here is required, not defensive. Matches what the RoboTwin client
    does before handing poses to the simulator (`eval_polict_client_openpi.py:616-621`).
    """
    quat = np.asarray(pose7[3:7], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"degenerate quaternion in predicted pose: {pose7[3:7]}")
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat(quat / norm).as_matrix()
    mat[:3, 3] = np.asarray(pose7[:3], dtype=np.float64)
    return mat


def joints14_to_eef16(joints14: np.ndarray, kin_left: YamKinematics,
                      kin_right: YamKinematics) -> np.ndarray:
    """(14,) YAM joints -> (16,) absolute EEF pose, the model's layout.

    The same FK `script/convert_yam_to_lingbot_va.py` applies to the training data, so
    an anchor built with this is in exactly the space the model's deltas expect.
    """
    out = np.zeros(16, dtype=np.float64)
    for kin, q_slice, grip_idx, dst in (
        (kin_left, slice(0, ARM_DOF), ARM_DOF, LEFT_ARM),
        (kin_right, slice(YAM_DOF, YAM_DOF + ARM_DOF), YAM_DOF + ARM_DOF, RIGHT_ARM),
    ):
        mat = kin.fk(joints14[q_slice])
        arm = np.empty(8, dtype=np.float64)
        arm[:3] = mat[:3, 3]
        arm[3:7] = R.from_matrix(mat[:3, :3]).as_quat()
        arm[7] = joints14[grip_idx]
        out[dst] = arm
    return out


class YamActionConverter:
    """Stateful converter from model action chunks to YAM joint commands.

    Two pieces of state make this stateful rather than a pure function, and both
    matter for correctness:

    - **The anchor**, latched once per rollout by `latch_anchor`. The model's poses are
      relative to the configuration the rollout started from, so the anchor must be
      re-latched exactly when the server's KV cache is reset -- never mid-rollout.
    - **The IK seed**, carried forward across steps. This solver is differential and
      returns whichever branch it converges to, so seeding from the previous command
      is what keeps the elbow from flipping between adjacent actions.
    """

    action_space = "eef"
    action_dim = 16   # channels of postprocess_action's output

    def __init__(self, i2rt_root: str = DEFAULT_I2RT_ROOT, clip_gripper: bool = True,
                 max_jump_rad: float = 0.2):
        # Two instances so each arm's IK seed evolves independently. They load the same
        # XML; the separation is about solver state, not about the models differing.
        self.kin_left = YamKinematics(i2rt_root)
        self.kin_right = YamKinematics(i2rt_root)
        self.clip_gripper = clip_gripper
        # Escaping a stalled IK seed means switching solution branch, which shows up as
        # a step change in the joint command. Anything above this is reported in
        # `stats["unsafe_jumps"]`: a live client MUST gate on that rather than execute
        # the slew, because `_check_joint_delta` compares the command against measured
        # state and calls `emergency_stop()` on exceedance (`raiden/server.py:944-980`).
        self.max_jump_rad = max_jump_rad
        self.anchor16 = None
        self._seed_left = None
        self._seed_right = None
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "steps": 0,
            "ik_failures": 0,
            "ik_fallbacks": 0,
            "max_fallback_jump_rad": 0.0,
            "unsafe_jumps": 0,
            "gripper_clipped": 0,
            "max_pos_residual_m": 0.0,
            "max_ori_residual_rad": 0.0,
        }

    def latch_anchor(self, joints14: np.ndarray) -> np.ndarray:
        """Latch the rollout anchor from measured joints, and seed IK from them.

        Call once per rollout, at the same moment the policy server's KV cache is
        reset. Training relativizes each segment against its own first frame, and
        `script/convert_yam_to_lingbot_va.py --chunk-frames 0` emits one segment per
        episode, so the training anchor is episode frame 0 -- matching a deployment
        that latches once and never re-anchors.
        """
        joints14 = np.asarray(joints14, dtype=np.float64).reshape(-1)
        if joints14.shape[0] != BIMANUAL_DOF:
            raise ValueError(f"expected (14,) joints, got {joints14.shape}")
        self.anchor16 = joints14_to_eef16(joints14, self.kin_left, self.kin_right)
        self._seed_left = joints14[:ARM_DOF].copy()
        self._seed_right = joints14[YAM_DOF:YAM_DOF + ARM_DOF].copy()
        return self.anchor16

    def convert_step(self, action16: np.ndarray) -> np.ndarray:
        """One 16-dim relative EEF action -> (14,) YAM joint command."""
        if self.anchor16 is None:
            raise RuntimeError("latch_anchor() must be called before convert_step()")
        action16 = np.asarray(action16, dtype=np.float64).reshape(-1)
        if action16.shape[0] != 16:
            raise ValueError(f"expected (16,) action, got {action16.shape}")

        # Validate before composing, not after. `pose_to_matrix` has the degenerate
        # check, but `add_eef_pose` runs first and calls `R.from_quat` on the raw
        # model output -- scipy normalizes a non-unit quaternion silently but raises
        # "Found zero norm quaternions" on a zero one, so the useful diagnostic never
        # got a chance to fire. Checking here covers both arms at the boundary where
        # model output enters.
        for name, quat in (("left", action16[3:7]), ("right", action16[11:15])):
            norm = float(np.linalg.norm(quat))
            if not np.isfinite(norm) or norm < 1e-8:
                raise ValueError(
                    f"degenerate {name} quaternion in predicted action: {quat}. The "
                    "model emits unnormalized regression targets, but a zero-norm one "
                    "means the output is garbage -- check the norm stats and config.")

        absolute16 = add_init_pose(action16, self.anchor16)
        out = np.zeros(BIMANUAL_DOF, dtype=np.float32)

        for kin, src, seed_attr, dst_q, dst_grip in (
            (self.kin_left, LEFT_ARM, "_seed_left", slice(0, ARM_DOF), ARM_DOF),
            (self.kin_right, RIGHT_ARM, "_seed_right",
             slice(YAM_DOF, YAM_DOF + ARM_DOF), YAM_DOF + ARM_DOF),
        ):
            arm = absolute16[src]
            target = pose_to_matrix(arm[POSE])
            ok, q6, used_fallback = kin.ik_robust(
                target, seed_q6=getattr(self, seed_attr))
            if not ok:
                self.stats["ik_failures"] += 1
            if used_fallback:
                self.stats["ik_fallbacks"] += 1
                previous = getattr(self, seed_attr)
                jump = float(np.abs(q6 - previous).max()) if previous is not None else 0.0
                self.stats["max_fallback_jump_rad"] = max(
                    self.stats["max_fallback_jump_rad"], jump)
                if jump > self.max_jump_rad:
                    self.stats["unsafe_jumps"] += 1
            setattr(self, seed_attr, q6)

            # Round-trip residual, position *and* orientation. The model is fed back
            # its own emitted action rather than measured state
            # (`wan_va_server.py:583`), so IK error never reaches the policy --
            # tracking it here is the only way it becomes visible. Orientation is
            # tracked separately because a solution can sit at the position threshold
            # while being rotated away from the target.
            achieved = kin.fk(q6)
            self.stats["max_pos_residual_m"] = max(
                self.stats["max_pos_residual_m"],
                float(np.linalg.norm(achieved[:3, 3] - target[:3, 3])))
            self.stats["max_ori_residual_rad"] = max(
                self.stats["max_ori_residual_rad"],
                float((R.from_matrix(achieved[:3, :3]).inv()
                       * R.from_matrix(target[:3, :3])).magnitude()))

            grip = float(arm[GRIPPER])
            if self.clip_gripper:
                clipped = min(max(grip, 0.0), 1.0)
                if clipped != grip:
                    self.stats["gripper_clipped"] += 1
                grip = clipped

            out[dst_q] = q6
            out[dst_grip] = grip

        self.stats["steps"] += 1
        return out

    def convert_sequence(self, actions_t16: np.ndarray) -> np.ndarray:
        """(T, 16) relative EEF actions -> (T, 14) joint commands."""
        actions_t16 = np.asarray(actions_t16, dtype=np.float64)
        if actions_t16.ndim != 2 or actions_t16.shape[1] != 16:
            raise ValueError(f"expected (T, 16), got {actions_t16.shape}")
        return np.stack([self.convert_step(a) for a in actions_t16])

    def convert_chunk(self, action_cfn: np.ndarray) -> np.ndarray:
        """`postprocess_action` output (16, F, N) -> (F*N, 14) in execution order.

        The (frame, action-within-frame) flattening order matches how the RoboTwin
        client steps a chunk (`eval_polict_client_openpi.py:600-602`).
        """
        action_cfn = np.asarray(action_cfn, dtype=np.float64)
        if action_cfn.ndim != 3 or action_cfn.shape[0] != 16:
            raise ValueError(f"expected (16, F, N), got {action_cfn.shape}")
        return self.convert_sequence(action_cfn.transpose(1, 2, 0).reshape(-1, 16))

    def predicted_rel16(self, action_cfn: np.ndarray) -> np.ndarray:
        """(16, F, N) model output -> (F*N, 16) anchor-relative EEF, in execution order.

        What the model asked for, in the frame the plots and EEF metrics use. For this
        converter that is the model output itself.
        """
        return np.asarray(action_cfn, dtype=np.float64).transpose(1, 2, 0).reshape(-1, 16)


class YamJointActionConverter:
    """Model action chunks -> YAM joint commands, for `action_space == 'joint'` checkpoints.

    The model already emits absolute `[left q6, left grip, right q6, right grip]`, the
    layout raiden's `--action-type joint` takes, so there is no IK, no seed and no
    branch to switch. What is left is clipping: arm joints to the limits the hardware
    enforces (it would clip them silently otherwise, and the plot would show motion
    the arm never made), grippers to [0, 1].

    Same interface as `YamActionConverter`, so callers need not branch. The anchor is
    still latched, but only to express predictions as anchor-relative EEF for the
    plots (`predicted_rel16`) -- nothing about the commands depends on it. The IK
    entries of `stats` stay 0; `unsafe_jumps` in particular is an IK branch switch and
    cannot happen here. Large steps are left to the client's slew limiter.
    """

    action_space = "joint"
    action_dim = BIMANUAL_DOF

    def __init__(self, i2rt_root: str = DEFAULT_I2RT_ROOT, clip_gripper: bool = True,
                 max_jump_rad: float = 0.2):
        # FK only: the anchor and predicted_rel16 for plots, and the joint limits.
        self.kin_left = YamKinematics(i2rt_root)
        self.kin_right = YamKinematics(i2rt_root)
        self.clip_gripper = clip_gripper
        self.max_jump_rad = max_jump_rad   # unused; kept so callers can print it
        # Both arms load the same XML, so one range serves both.
        self.joint_range = self.kin_left.arm_joint_range()
        self.anchor16 = None
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "steps": 0,
            "ik_failures": 0,
            "ik_fallbacks": 0,
            "max_fallback_jump_rad": 0.0,
            "unsafe_jumps": 0,
            "gripper_clipped": 0,
            "max_pos_residual_m": 0.0,
            "max_ori_residual_rad": 0.0,
            # Steps with any arm joint outside joint_range, and the worst overshoot.
            "joint_limit_clipped": 0,
            "max_joint_overshoot_rad": 0.0,
        }

    def latch_anchor(self, joints14: np.ndarray) -> np.ndarray:
        """Latch the anchor used by `predicted_rel16`. Commands do not depend on it."""
        joints14 = np.asarray(joints14, dtype=np.float64).reshape(-1)
        if joints14.shape[0] != BIMANUAL_DOF:
            raise ValueError(f"expected (14,) joints, got {joints14.shape}")
        self.anchor16 = joints14_to_eef16(joints14, self.kin_left, self.kin_right)
        return self.anchor16

    def convert_step(self, action14: np.ndarray) -> np.ndarray:
        """One 14-dim absolute joint action -> (14,) clipped YAM joint command."""
        action14 = np.asarray(action14, dtype=np.float64).reshape(-1)
        if action14.shape[0] != BIMANUAL_DOF:
            raise ValueError(f"expected (14,) action, got {action14.shape}")
        if not np.all(np.isfinite(action14)):
            raise ValueError(f"non-finite joint action from the model: {action14}. "
                             "Check the norm stats and config.")
        out = action14.astype(np.float32)
        lo, hi = self.joint_range[:, 0], self.joint_range[:, 1]
        clipped_any = False
        for q_slice, grip_idx in ((slice(0, ARM_DOF), ARM_DOF),
                                  (slice(YAM_DOF, YAM_DOF + ARM_DOF), YAM_DOF + ARM_DOF)):
            q = action14[q_slice]
            overshoot = float(np.max(np.maximum(lo - q, q - hi)))
            if overshoot > 0:
                clipped_any = True
                self.stats["max_joint_overshoot_rad"] = max(
                    self.stats["max_joint_overshoot_rad"], overshoot)
            out[q_slice] = np.clip(q, lo, hi)

            grip = float(action14[grip_idx])
            if self.clip_gripper:
                clipped = min(max(grip, 0.0), 1.0)
                if clipped != grip:
                    self.stats["gripper_clipped"] += 1
                grip = clipped
            out[grip_idx] = grip
        self.stats["joint_limit_clipped"] += int(clipped_any)
        self.stats["steps"] += 1
        return out

    def convert_sequence(self, actions_t14: np.ndarray) -> np.ndarray:
        """(T, 14) absolute joint actions -> (T, 14) joint commands."""
        actions_t14 = np.asarray(actions_t14, dtype=np.float64)
        if actions_t14.ndim != 2 or actions_t14.shape[1] != BIMANUAL_DOF:
            raise ValueError(f"expected (T, 14), got {actions_t14.shape}")
        return np.stack([self.convert_step(a) for a in actions_t14])

    def convert_chunk(self, action_cfn: np.ndarray) -> np.ndarray:
        """`postprocess_action` output (14, F, N) -> (F*N, 14) in execution order."""
        action_cfn = np.asarray(action_cfn, dtype=np.float64)
        if action_cfn.ndim != 3 or action_cfn.shape[0] != BIMANUAL_DOF:
            raise ValueError(f"expected (14, F, N), got {action_cfn.shape}")
        return self.convert_sequence(
            action_cfn.transpose(1, 2, 0).reshape(-1, BIMANUAL_DOF))

    def predicted_rel16(self, action_cfn: np.ndarray) -> np.ndarray:
        """(14, F, N) model output -> (F*N, 16) anchor-relative EEF, via FK.

        FK of the joints the model asked for, before clipping -- the EEF counterpart of
        `YamActionConverter.predicted_rel16`, so an EEF and a joint checkpoint plot and
        score in one frame.
        """
        if self.anchor16 is None:
            raise RuntimeError("latch_anchor() must be called before predicted_rel16()")
        joints = np.asarray(action_cfn, dtype=np.float64).transpose(1, 2, 0).reshape(
            -1, BIMANUAL_DOF)
        abs16 = np.stack([joints14_to_eef16(q, self.kin_left, self.kin_right)
                          for q in joints])
        return relativize_action16(np.concatenate([self.anchor16[None], abs16]))[1:]


def make_converter(action_space: str, i2rt_root: str = DEFAULT_I2RT_ROOT, **kwargs):
    """The converter for a checkpoint's `action_space` ('eef' or 'joint')."""
    classes = {"eef": YamActionConverter, "joint": YamJointActionConverter}
    if action_space not in classes:
        raise ValueError(f"unknown action_space {action_space!r}; expected one of "
                         f"{sorted(classes)}")
    return classes[action_space](i2rt_root, **kwargs)
