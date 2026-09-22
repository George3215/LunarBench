#!/usr/bin/env python3
"""`stone_stack/robots/kinematics.py` 的离线测试：迁移自 GPT-as-Policy 的几何工具。

不导入 mujoco（`preview_trajectory` 用假的 controller 桩），纯几何/契约校验。

    python tools/test_kinematics_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from stone_stack.robots import kinematics as K  # noqa: E402

CHECKS = {"passed": 0, "failed": 0}


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        CHECKS["passed"] += 1
    else:
        CHECKS["failed"] += 1
        print(f"  FAIL {name}: {detail}")


def raises(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except (ValueError, KeyError):
        return True
    return False


def quat(axis, angle_deg) -> np.ndarray:
    q = Rotation.from_rotvec(np.asarray(axis, float) * np.deg2rad(angle_deg)).as_quat()
    return q[[3, 0, 1, 2]]


def main() -> int:
    print("[1] 位姿互转")
    matrix = K.transform([0.1, 0.2, 0.3], quat([0, 0, 1], 30))
    back = K.pose(matrix)
    check("position 往返", np.allclose(back["position"], [0.1, 0.2, 0.3], atol=1e-9))
    check("quaternion 往返", np.allclose(back["quaternion_wxyz"], quat([0, 0, 1], 30), atol=1e-9))
    check("非单位四元数被拒", raises(K.transform, [0, 0, 0], [2.0, 0, 0, 0]))
    check("矩阵↔四元数一致",
          np.allclose(K.quaternion_to_matrix(K.matrix_to_quaternion_wxyz(matrix[:3, :3])), matrix[:3, :3]))

    print("[2] 任务空间限幅（迁移自 ArmFK.bounded_target）")
    current = np.zeros(3)
    diagnostic = K.bounded_task_error(current, np.eye(3), np.array([0.5, 0.0, 0.0]), np.eye(3))
    check("位移被限到 2 cm", np.isclose(np.linalg.norm(diagnostic["bounded_translation"]), 0.02, atol=1e-9),
          str(diagnostic["bounded_translation"]))
    check("记录了原始误差", np.isclose(diagnostic["translation_error_m"], 0.5, atol=1e-9))
    check("clipped 标记为真", diagnostic["translation_clipped"] is True)
    rot_away = Rotation.from_rotvec([0, 0, 1.0]).as_matrix()
    diag_rot = K.bounded_task_error(current, np.eye(3), current, rot_away)
    check("旋转被限到 0.1 rad", np.isclose(diag_rot["rotation_error_rad"], 1.0, atol=1e-9)
          and np.isclose(np.linalg.norm(diag_rot["bounded_rotation_vector"]), 0.10, atol=1e-9))
    pos, rot, _ = K.bounded_eef_target(current, np.eye(3), np.array([0.5, 0, 0]), np.eye(3))
    check("限幅后的目标离当前 2 cm", np.isclose(np.linalg.norm(pos), 0.02, atol=1e-9))
    check("小误差不触发限幅",
          not K.bounded_task_error(current, np.eye(3), np.array([0.001, 0, 0]), np.eye(3))["translation_clipped"])

    print("[3] edited_targets（有界平滑修正）")
    trajectory = [dict(index=i, position=[0.2, 0.0, 0.1], quaternion_wxyz=[1.0, 0, 0, 0], gripper_closed=False)
                  for i in range(5)]
    edit = {"delta_position": [0.03, 0, 0], "delta_rotation_vector": [0, 0, 0.2], "gripper": "closed"}
    edited = K.edited_targets(trajectory, 4, edit)
    check("返回 4 步", len(edited) == 4)
    check("首步是 1/4 修正", np.isclose(edited[0]["position"][0], 0.2 + 0.03 / 4, atol=1e-9), str(edited[0]["position"]))
    check("末步用满修正", np.isclose(edited[3]["position"][0], 0.2 + 0.03, atol=1e-9))
    check("夹爪被覆盖为 closed", all(item["gripper_closed"] for item in edited))
    keep = K.edited_targets(trajectory, 2, {"delta_position": [0, 0, 0], "delta_rotation_vector": [0, 0, 0], "gripper": "keep"})
    check("keep 沿用原夹爪", all(item["gripper_closed"] is False for item in keep))
    check("超 5 cm 被拒", raises(K.edited_targets, trajectory, 3,
                                  {"delta_position": [0.06, 0, 0], "delta_rotation_vector": [0, 0, 0], "gripper": "keep"}))
    check("超 0.35 rad 被拒", raises(K.edited_targets, trajectory, 3,
                                     {"delta_position": [0, 0, 0], "delta_rotation_vector": [0, 0, 0.4], "gripper": "keep"}))
    check("步数 0 被拒", raises(K.edited_targets, trajectory, 0, edit))
    check("步数 16 被拒", raises(K.edited_targets, trajectory, 16, edit))

    print("[4] validate_eef_target")
    current_eef = {"position": [0.2, 0.0, 0.1], "quaternion_wxyz": [1.0, 0, 0, 0]}
    K.validate_eef_target({"position": [0.23, 0.0, 0.1], "quaternion_wxyz": [1.0, 0, 0, 0],
                           "gripper_closed": True}, current_eef)
    check("5 cm 内通过", True)
    check("超 5 cm 被拒", raises(K.validate_eef_target,
                                {"position": [0.3, 0.0, 0.1], "quaternion_wxyz": [1.0, 0, 0, 0], "gripper_closed": True},
                                current_eef))
    check("非单位四元数被拒", raises(K.validate_eef_target,
                                    {"position": [0.2, 0, 0.1], "quaternion_wxyz": [2.0, 0, 0, 0], "gripper_closed": True},
                                    current_eef))
    check("非布尔夹爪被拒", raises(K.validate_eef_target,
                                  {"position": [0.2, 0, 0.1], "quaternion_wxyz": [1.0, 0, 0, 0], "gripper_closed": 1},
                                  current_eef))

    print("[5] validate_execution_contract")
    request = {"request_id": "r1", "current_eef": current_eef,
               "student_eef_trajectory": [dict(position=[0.2, 0, 0.1], quaternion_wxyz=[1.0, 0, 0, 0], gripper_closed=False)
                                          for _ in range(5)]}
    K.validate_execution_contract({"request_id": "r1", "mode": "stop", "reason": "done"}, request)
    check("stop 通过", True)
    K.validate_execution_contract(
        {"request_id": "r1", "mode": "eef", "steps": 3, "reason": "fix",
         "target": {"position": [0.22, 0, 0.1], "quaternion_wxyz": [1.0, 0, 0, 0], "gripper_closed": False}}, request)
    check("eef 通过", True)
    K.validate_execution_contract(
        {"request_id": "r1", "mode": "edit", "steps": 3, "reason": "shift",
         "edit": {"delta_position": [0.01, 0, 0], "delta_rotation_vector": [0, 0, 0], "gripper": "keep"}}, request)
    check("edit 通过", True)
    check("陈旧 request_id 被拒",
          raises(K.validate_execution_contract, {"request_id": "r0", "mode": "stop", "reason": "x"}, request))
    check("未知 mode 被拒",
          raises(K.validate_execution_contract, {"request_id": "r1", "mode": "teleport", "reason": "x"}, request))
    check("空 reason 被拒",
          raises(K.validate_execution_contract, {"request_id": "r1", "mode": "stop", "reason": "  "}, request))
    check("student 步数 16 被拒",
          raises(K.validate_execution_contract, {"request_id": "r1", "mode": "student", "steps": 16, "reason": "x"}, request))
    check("edit 步数 6 被拒",
          raises(K.validate_execution_contract,
                 {"request_id": "r1", "mode": "edit", "steps": 6, "reason": "x",
                  "edit": {"delta_position": [0, 0, 0], "delta_rotation_vector": [0, 0, 0], "gripper": "keep"}}, request))

    print("[6] validate_public_language")
    K.validate_public_language({"reason": "move left slightly", "assessment": {}})
    check("英文通过", True)
    check("CJK 被拒", raises(K.validate_public_language, {"reason": "往左移一点", "assessment": {}}))

    print("[7] preview_trajectory（假 controller 桩）")

    class FakeController:
        def tcp_pose_for(self, q):
            q = np.asarray(q, float)
            return np.array([q[0], q[1], q[2]]), Rotation.from_rotvec([0, 0, q[0]]).as_matrix()

    preview = K.preview_trajectory(FakeController(), [[0.1, 0, 0.05], [0.2, 0, 0.05]], [False, True])
    check("返回 2 步", len(preview) == 2)
    check("位置来自 FK", np.allclose(preview[1]["position"], [0.2, 0, 0.05]))
    check("夹爪随步", preview[0]["gripper_closed"] is False and preview[1]["gripper_closed"] is True)

    total = CHECKS["passed"] + CHECKS["failed"]
    print(f"\nTOTAL {CHECKS['passed']}/{total} passed")
    return 0 if CHECKS["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
