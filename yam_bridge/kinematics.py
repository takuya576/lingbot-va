# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Shared YAM forward/inverse kinematics for the `grasp_site` end-effector frame.

Both directions of the YAM <-> LingBot-VA action conversion live here, on purpose:

  - **Training** (`script/convert_yam_to_lingbot_va.py`) needs FK, to turn recorded
    joint angles into the 16-dim bimanual EEF pose the model is trained on.
  - **Deployment** (`yam_bridge/action_convert.py`) needs IK, to turn the model's
    predicted EEF pose back into joint commands for the arms.

If those two used different MuJoCo models -- a different arm XML, a different gripper,
a different site -- the round trip would pick up a constant offset that no test on
either side alone would catch. Sharing one `YamKinematics` object makes that class of
bug unrepresentable.

`i2rt.robots.kinematics.Kinematics` is imported by path rather than as an installed
package (see `_import_kinematics_cls`), so this module works in any environment with
`mujoco` + `mink`, without installing i2rt or pulling in its CAN-bus driver chain.
"""
import importlib.util
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

ARM_DOF = 6
SITE_NAME = "grasp_site"
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, ARM_DOF + 1))

# i2rt widens the limits it enforces on hardware by this much before handing them to
# MotorChainRobot (`get_robot.py:131-133`, commented "safety buffer"), so the arm really
# does reach configurations the unmodified MuJoCo XML forbids.
ARM_LIMIT_BUFFER = 0.15

# Seeds retried when the carried seed stalls in a local minimum. Zero is the arm's
# neutral configuration and empirically escapes the stalls seen in cover_blocks; the
# second is a mildly folded pose, to give the retry a second basin to start from.
DEFAULT_FALLBACK_SEEDS = (
    np.zeros(ARM_DOF),
    np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
)


def combine_arm_and_gripper_xml(arm_path: str, gripper_path: str) -> str:
    """Reimplementation of i2rt.robots.utils.combine_arm_and_gripper_xml.

    Vendored (rather than importing i2rt.robots.utils) to avoid pulling in i2rt's
    CAN-bus hardware driver import chain, which pure kinematics does not need.
    Mirrors `raiden._xml_paths.get_yam_4310_linear_xml_path`, which combines the same
    two XMLs -- keep them in sync.
    """
    arm_tree = ET.parse(arm_path)
    arm_root = arm_tree.getroot()
    arm_dir = os.path.dirname(os.path.abspath(arm_path))
    arm_compiler = arm_root.find("compiler")
    arm_meshdir = arm_compiler.get("meshdir", "") if arm_compiler is not None else ""
    arm_asset = arm_root.find("asset")
    for child in arm_asset:
        if child.get("file") and not os.path.isabs(child.get("file")):
            child.set("file", os.path.abspath(os.path.join(arm_dir, arm_meshdir, child.get("file"))))
    if arm_compiler is not None and arm_compiler.get("meshdir"):
        del arm_compiler.attrib["meshdir"]

    grip_tree = ET.parse(gripper_path)
    grip_root = grip_tree.getroot()
    grip_body = grip_root.find(".//body[@name='link_6']")
    if grip_body is None:
        grip_body = grip_root.find(".//body[@name='link6']")
    grip_dir = os.path.dirname(os.path.abspath(gripper_path))
    grip_compiler = grip_root.find("compiler")
    grip_meshdir = grip_compiler.get("meshdir", "") if grip_compiler is not None else ""
    grip_asset = grip_root.find("asset")
    if grip_asset is not None:
        existing = {(c.tag, c.get("name")) for c in arm_asset}
        for child in grip_asset:
            key = (child.tag, child.get("name"))
            if key not in existing:
                elem = deepcopy(child)
                if elem.get("file") and not os.path.isabs(elem.get("file")):
                    elem.set("file", os.path.abspath(os.path.join(grip_dir, grip_meshdir, elem.get("file"))))
                arm_asset.append(elem)
                existing.add(key)
    if grip_body is not None:
        for parent in arm_root.iter():
            for idx, child in enumerate(list(parent)):
                if child.tag == "body" and child.get("name") in ("link_6", "link6"):
                    parent.remove(child)
                    parent.insert(idx, deepcopy(grip_body))

    fd, path = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    arm_tree.write(path)
    return path


def widen_arm_joint_limits(xml_path: str, buffer: float = ARM_LIMIT_BUFFER) -> str:
    """Widen the six arm joints' ranges by `buffer` rad in place; return the path.

    Required for IK to match the hardware. mink enforces the model's configuration
    limits by default -- `solve_ik` substitutes `[ConfigurationLimit(model)]` whenever
    `limits is None` (`mink/solve_ik.py:29-30`) -- and the stock XML ranges are exactly
    `_ARM_JOINT_LIMITS[ArmType.YAM]`, i.e. the *un*buffered table. Recorded cover_blocks
    joint 4 reaches -1.715 against a stock limit of -1.571, so without this the solver
    saturates at the limit, returns a pose ~2 cm off, then seeds the next step from that
    wrong configuration and never recovers.

    Gripper joints are deliberately left alone: they do not move `grasp_site`.
    """
    tree = ET.parse(xml_path)
    for joint in tree.getroot().iter("joint"):
        rng = joint.get("range")
        if rng is None or joint.get("name", "") not in ARM_JOINT_NAMES:
            continue
        lo, hi = (float(v) for v in rng.split())
        joint.set("range", f"{lo - buffer} {hi + buffer}")
    tree.write(xml_path)
    return xml_path


def yam_4310_linear_xml(i2rt_root: str, limit_buffer: float = ARM_LIMIT_BUFFER) -> str:
    """Path to a combined YAM arm + linear_4310 gripper XML (written to a temp file).

    `limit_buffer` widens the arm joint ranges to match what the hardware allows; pass
    0.0 for the stock model.
    """
    arm_xml = os.path.join(i2rt_root, "robot_models/arm/yam/yam.xml")
    gripper_xml = os.path.join(i2rt_root, "robot_models/gripper/linear_4310/linear_4310.xml")
    combined = combine_arm_and_gripper_xml(arm_xml, gripper_xml)
    if limit_buffer:
        widen_arm_joint_limits(combined, limit_buffer)
    return combined


def _import_kinematics_cls(i2rt_root: str):
    """Import `i2rt.robots.kinematics.Kinematics` from a source tree, uninstalled.

    `i2rt/robots/` has no `__init__.py` and `kinematics.py` imports only mink, mujoco
    and numpy, so loading it directly by file path costs nothing and avoids requiring
    i2rt to be installed in the caller's environment.
    """
    path = os.path.join(i2rt_root, "robots", "kinematics.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"i2rt kinematics not found at {path}. Pass --i2rt-root pointing at the "
            "i2rt package directory, e.g. /data2/takuya/raiden/third_party/i2rt/i2rt"
        )
    spec = importlib.util.spec_from_file_location("_i2rt_kinematics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_i2rt_kinematics", module)
    spec.loader.exec_module(module)
    return module.Kinematics


class YamKinematics:
    """FK and IK for one YAM arm's `grasp_site`, in that arm's own base frame.

    Poses are always expressed in the arm's own base frame -- the same convention as
    raiden's on-disk actions and live action space (`raiden/server.py:800-805`) and as
    `script/convert_yam_to_lingbot_va.py`'s FK. There is deliberately no notion of a
    shared world frame here, so the left and right arms use two independent instances.
    """

    def __init__(self, i2rt_root: str, limit_buffer: float = ARM_LIMIT_BUFFER):
        kinematics_cls = _import_kinematics_cls(i2rt_root)
        self._xml_path = yam_4310_linear_xml(i2rt_root, limit_buffer)
        self._kin = kinematics_cls(self._xml_path, SITE_NAME)
        self.nq = self._kin._configuration.model.nq

    def _pad(self, q6: np.ndarray) -> np.ndarray:
        """Widen 6 arm joints to the model's full nq; gripper joints stay at zero.

        The gripper's finger joints do not move `grasp_site`, so zeroing them is exact
        rather than approximate -- FK and IK agree regardless of gripper opening.
        """
        q = np.zeros(self.nq, dtype=np.float64)
        q[:ARM_DOF] = np.asarray(q6, dtype=np.float64)[:ARM_DOF]
        return q

    def fk(self, q6: np.ndarray) -> np.ndarray:
        """6 arm joints (rad) -> 4x4 transform of `grasp_site` in the arm base frame."""
        return self._kin.fk(self._pad(q6))

    def ik_robust(self, target_pose: np.ndarray, seed_q6: Optional[np.ndarray] = None,
                  fallback_seeds: Tuple[np.ndarray, ...] = None,
                  **kwargs) -> Tuple[bool, np.ndarray, bool]:
        """`ik`, retried from alternative seeds when the primary seed stalls.

        Returns `(converged, q6, used_fallback)`.

        The differential solver can land in a local minimum from which no amount of
        extra iteration escapes -- measured on cover_blocks episode 5, a seed that had
        drifted 0.73 rad from the demonstrated configuration stayed 3.2e-4 m off the
        target after 10000 iterations (2.9 s), while the same target solved from a
        neutral seed in 4.2 ms. Retrying is therefore the only fix; raising `max_iters`
        is not.

        Among the fallbacks that do converge, the one nearest the original seed wins,
        so the commanded trajectory stays as continuous as the escape allows -- a jump
        here is what `_check_joint_delta` e-stops on (`raiden/server.py:944-980`).
        """
        ok, q = self.ik(target_pose, seed_q6, **kwargs)
        if ok:
            return True, q, False

        seeds = DEFAULT_FALLBACK_SEEDS if fallback_seeds is None else fallback_seeds
        reference = np.asarray(seed_q6, dtype=np.float64)[:ARM_DOF] if seed_q6 is not None else None
        best = None
        for candidate in seeds:
            ok_fb, q_fb = self.ik(target_pose, candidate, **kwargs)
            if not ok_fb:
                continue
            distance = 0.0 if reference is None else float(np.abs(q_fb - reference).max())
            if best is None or distance < best[0]:
                best = (distance, q_fb)
        if best is not None:
            return True, best[1], True
        return False, q, False

    def ik(self, target_pose: np.ndarray, seed_q6: Optional[np.ndarray] = None,
           **kwargs) -> Tuple[bool, np.ndarray]:
        """4x4 target -> (converged, 6 arm joints).

        Seed with the previous *commanded* configuration when solving a trajectory:
        this differential solver returns whichever branch it converges to, so an
        unseeded call can jump between elbow configurations between adjacent steps.

        Note the solver does not enforce joint limits unless `limits=` is passed, and
        `MotorChainRobot.command_joint_pos` silently clips out-of-range commands -- so
        treat `converged=False` as a fault rather than a warning.
        """
        init_q = self._pad(seed_q6) if seed_q6 is not None else None
        ok, q = self._kin.ik(target_pose, SITE_NAME, init_q=init_q, **kwargs)
        return bool(ok), np.asarray(q[:ARM_DOF], dtype=np.float64)


def build_fk(i2rt_root: str):
    """Return `fk(q6) -> 4x4` for the YAM `grasp_site`.

    Thin wrapper kept for `script/convert_yam_to_lingbot_va.py`, which only needs FK.
    """
    return YamKinematics(i2rt_root).fk
