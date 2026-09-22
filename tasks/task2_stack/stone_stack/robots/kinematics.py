"""机器人几何工具：位姿转换、有界修正、任务空间限幅、执行契约校验。

**来源与定位**：本模块是从 `policy/GPT-as-Policy` 的
`robodojo_server/action_edit_kinematics.py`、`robodojo_server/validation.py`
（以及 `kinematics.py` 的限幅思想）**移植并单臂适配**到 TASK2 的版本。

它只做纯几何/契约校验，不导入 mujoco、不碰仿真状态：

- `transform` / `pose` / `quaternion_to_matrix` / `matrix_to_quaternion_wxyz`：
  四元数（w x y z）与旋转矩阵互转，与 GPT-as-Policy 的落盘字段保持一致。
- `edited_targets`：对一段 EEF 轨迹做**有界平滑修正**（位移 ≤5 cm、旋转 ≤0.35 rad），
  修正量按步序号线性展开（`alpha = (i+1)/steps`），并支持夹爪覆盖。
- `bounded_task_error` / `bounded_eef_target`：GPT-as-Policy `ArmFK.bounded_target`
  里"任务空间误差限幅"的移植（默认 2 cm / 0.1 rad）——这是 task2 原来的
  `solve_ik` **没有**的安全层：它约束的是*这一步允许走多远*，而不是最终解的质量。
- `validate_eef_target`：显式 EEF 目标必须在当前 EEF 的 5 cm / 0.35 rad 内、
  四元数单位化、夹爪必须显式布尔。
- `validate_execution_contract`：`student / edit / eef / stop` 四模式契约、
  步数上限（student 15 / edit·eef 5）、`request_id` 防陈旧。
- `validate_public_language`：公开解释文本禁止 CJK（沿用 GPT-as-Policy 约定）。

**未迁移**的部分与原因见 `docs/GPT_TOOLS_MIGRATION.md`。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "MAX_EDIT_TRANSLATION_M",
    "MAX_EDIT_ROTATION_RAD",
    "BOUNDED_TASK_TRANSLATION_M",
    "BOUNDED_TASK_ROTATION_RAD",
    "quaternion_to_matrix",
    "matrix_to_quaternion_wxyz",
    "transform",
    "pose",
    "pose_from_arrays",
    "bounded_task_error",
    "bounded_eef_target",
    "edited_targets",
    "preview_trajectory",
    "validate_eef_target",
    "validate_public_language",
    "validate_execution_contract",
]

#: 有界修正（edit）的硬上限：GPT-as-Policy 的短修正契约。
MAX_EDIT_TRANSLATION_M = 0.05
MAX_EDIT_ROTATION_RAD = 0.35

#: 单步任务空间限幅（`ArmFK.bounded_target` 的移植值）。
BOUNDED_TASK_TRANSLATION_M = 0.02
BOUNDED_TASK_ROTATION_RAD = 0.10

#: 各模式允许的最大执行步数（GPT-as-Policy `validate_response` 的契约）。
_STEP_CAP = {"student": 15, "edit": 5, "eef": 5}


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{label} 必须是 {size} 维有限向量，实际 {value!r}")
    return array


# --------------------------------------------------------------------------------------
# 位姿互转
# --------------------------------------------------------------------------------------


def quaternion_to_matrix(quaternion_wxyz: Any) -> np.ndarray:
    """单位四元数 (w x y z) → 3x3 旋转矩阵。"""
    quaternion = _finite_vector(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f"四元数必须单位化（|q|={norm:.6f}）")
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def matrix_to_quaternion_wxyz(rotation: Any) -> np.ndarray:
    """3x3 旋转矩阵 → 单位四元数 (w x y z)。"""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation 必须是 3x3 有限矩阵")
    quat = Rotation.from_matrix(matrix).as_quat()  # x y z w
    return quat[[3, 0, 1, 2]]


def transform(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    """位置 + 四元数 (w x y z) → 4x4 齐次变换。"""
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix(quaternion_wxyz)
    matrix[:3, 3] = _finite_vector(position, 3, "position")
    return matrix


def pose(matrix: Any) -> dict[str, list[float]]:
    """4x4 齐次变换 → `{position, quaternion_wxyz}`（GPT-as-Policy 的落盘字段名）。"""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("matrix 必须是 4x4 有限矩阵")
    return {
        "position": matrix[:3, 3].tolist(),
        "quaternion_wxyz": matrix_to_quaternion_wxyz(matrix[:3, :3]).tolist(),
    }


# --------------------------------------------------------------------------------------
# 任务空间限幅
# --------------------------------------------------------------------------------------


def bounded_task_error(
    current_pos: Any,
    current_rot: Any,
    target_pos: Any,
    target_rot: Any,
    *,
    max_translation_m: float = BOUNDED_TASK_TRANSLATION_M,
    max_rotation_rad: float = BOUNDED_TASK_ROTATION_RAD,
) -> dict[str, Any]:
    """算当前→目标的位姿误差，并把**这一步允许的增量**限幅。

    返回 `translation`（原始）、`rotation_vector`（原始）、`bounded_translation`、
    `bounded_rotation_vector` 与是否触发限幅的布尔量。原始误差保留，便于上层记录
    "请求了多少、实际只批准多少"。
    """
    current_pos = _finite_vector(current_pos, 3, "current_pos")
    target_pos = _finite_vector(target_pos, 3, "target_pos")
    current_rot = np.asarray(current_rot, dtype=float)
    target_rot = np.asarray(target_rot, dtype=float)
    if current_rot.shape != (3, 3) or target_rot.shape != (3, 3):
        raise ValueError("current_rot / target_rot 必须是 3x3 旋转矩阵")
    translation = target_pos - current_pos
    rotation_vector = Rotation.from_matrix(target_rot @ current_rot.T).as_rotvec()

    def _cap(vector: np.ndarray, cap: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= cap or norm <= 1e-12:
            return vector
        return vector * (cap / norm)

    bounded_translation = _cap(translation, float(max_translation_m))
    bounded_rotation = _cap(rotation_vector, float(max_rotation_rad))
    return {
        "translation": translation,
        "rotation_vector": rotation_vector,
        "bounded_translation": bounded_translation,
        "bounded_rotation_vector": bounded_rotation,
        "translation_clipped": bool(np.linalg.norm(bounded_translation - translation) > 1e-12),
        "rotation_clipped": bool(np.linalg.norm(bounded_rotation - rotation_vector) > 1e-12),
        "translation_error_m": float(np.linalg.norm(translation)),
        "rotation_error_rad": float(np.linalg.norm(rotation_vector)),
        "limits": {"max_translation_m": float(max_translation_m),
                   "max_rotation_rad": float(max_rotation_rad)},
    }


def bounded_eef_target(
    current_pos: Any,
    current_rot: Any,
    target_pos: Any,
    target_rot: Any,
    *,
    max_translation_m: float = BOUNDED_TASK_TRANSLATION_M,
    max_rotation_rad: float = BOUNDED_TASK_ROTATION_RAD,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """把绝对 EEF 目标限幅成"离当前位置不超过一步"的目标，返回 (pos, rot, 诊断)。

    这就是 GPT-as-Policy `ArmFK.bounded_target` 的等价物：任务空间先限幅，再交给
    IK 解算。task2 的 `RobotController.solve_ik` 本身只裁关节限位，不限制这一步
    走多远，所以这层限幅要单独做。
    """
    diagnostics = bounded_task_error(
        current_pos, current_rot, target_pos, target_rot,
        max_translation_m=max_translation_m, max_rotation_rad=max_rotation_rad,
    )
    pos = _finite_vector(current_pos, 3, "current_pos") + diagnostics["bounded_translation"]
    delta_rot = Rotation.from_rotvec(diagnostics["bounded_rotation_vector"]).as_matrix()
    rot = delta_rot @ np.asarray(current_rot, dtype=float)
    return pos, rot, diagnostics


# --------------------------------------------------------------------------------------
# 运动学前瞻（DualKinematics.preview 的单臂版）
# --------------------------------------------------------------------------------------


def preview_trajectory(controller: Any, joint_trajectory: Sequence[Any], gripper_closed: Sequence[bool] | None = None) -> list[dict[str, Any]]:
    """对一段关节轨迹做 FK 前瞻，返回逐步的 EEF 位姿（不推进物理）。

    `controller` 只需提供 `tcp_pose_for(q)`（见 `RobotController`）。这是
    GPT-as-Policy `DualKinematics.preview` 的单臂移植：那边用 URDF 解析 FK，
    这里直接用 task2 已经加载好的 MuJoCo 模型，避免两套 FK 不一致。
    """
    poses: list[dict[str, Any]] = []
    for index, q in enumerate(joint_trajectory):
        position, rotation = controller.tcp_pose_for(np.asarray(q, dtype=float))
        item: dict[str, Any] = dict(index=index, **pose_from_arrays(position, rotation))
        if gripper_closed is not None:
            item["gripper_closed"] = bool(gripper_closed[index])
        poses.append(item)
    return poses


def pose_from_arrays(position: Any, rotation: Any) -> dict[str, list[float]]:
    """(位置, 3x3 旋转) → `{position, quaternion_wxyz}`。"""
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation, dtype=float)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return pose(matrix)


# --------------------------------------------------------------------------------------
# 有界轨迹修正（edited_targets）
# --------------------------------------------------------------------------------------


def edited_targets(trajectory: Sequence[Mapping[str, Any]], steps: int, edit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """对 EEF 轨迹前 `steps` 步做有界平滑修正。

    与 GPT-as-Policy `edited_targets` 契约一致：
    - `steps` 必须是 1..15 的整数；
    - `edit` 必须恰好含 `delta_position`(3,) / `delta_rotation_vector`(3,) / `gripper`；
    - 位移总量 ≤ 5 cm，旋转总量 ≤ 0.35 rad；
    - 第 i 步按 `(i+1)/steps` 线性展开（首步就有非零修正，末步用满修正量）；
    - `gripper` ∈ {keep, open, closed}，keep 沿用原步的 `gripper_closed`。
    """
    if type(steps) is not int or not 1 <= steps <= 15:
        raise ValueError("执行前缀步数必须是 1..15 的整数")
    if not isinstance(edit, Mapping) or set(edit) != {"delta_position", "delta_rotation_vector", "gripper"}:
        raise ValueError("edit 必须且只能含 delta_position / delta_rotation_vector / gripper")
    delta_position = _finite_vector(edit["delta_position"], 3, "delta_position")
    delta_rotation = _finite_vector(edit["delta_rotation_vector"], 3, "delta_rotation_vector")
    if float(np.linalg.norm(delta_position)) > MAX_EDIT_TRANSLATION_M + 1e-10:
        raise ValueError("修正位移超过 5 cm 短修正上限")
    if float(np.linalg.norm(delta_rotation)) > MAX_EDIT_ROTATION_RAD + 1e-10:
        raise ValueError("修正旋转超过 0.35 rad 短修正上限")
    if edit["gripper"] not in ("keep", "open", "closed"):
        raise ValueError("未知的夹爪覆盖值")

    targets: list[dict[str, Any]] = []
    for index, original in enumerate(list(trajectory)[:steps]):
        matrix = transform(original["position"], original["quaternion_wxyz"])
        alpha = (index + 1) / steps
        matrix[:3, 3] += alpha * delta_position
        matrix[:3, :3] = Rotation.from_rotvec(alpha * delta_rotation).as_matrix() @ matrix[:3, :3]
        if edit["gripper"] == "keep":
            gripper_closed = bool(original["gripper_closed"])
        else:
            gripper_closed = edit["gripper"] == "closed"
        targets.append(dict(index=index, **pose(matrix), gripper_closed=gripper_closed))
    return targets


# --------------------------------------------------------------------------------------
# 执行契约校验
# --------------------------------------------------------------------------------------


def validate_eef_target(
    target: Mapping[str, Any],
    current_eef: Mapping[str, Any],
    *,
    max_translation_m: float = MAX_EDIT_TRANSLATION_M,
    max_rotation_rad: float = MAX_EDIT_ROTATION_RAD,
) -> None:
    """显式 EEF 目标必须：有限、单位四元数、布尔夹爪，且在当前位置 5 cm / 0.35 rad 内。"""
    position = np.asarray(target["position"], dtype=float).reshape(-1)
    quaternion = np.asarray(target["quaternion_wxyz"], dtype=float).reshape(-1)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("EEF 目标需要 3 维位置与 4 维四元数")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("EEF 目标必须是有限位置/四元数")
    if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-4:
        raise ValueError("四元数必须单位化")
    if type(target["gripper_closed"]) is not bool:
        raise ValueError("夹爪必须是显式布尔值")

    current_position = np.asarray(current_eef["position"], dtype=float).reshape(-1)
    if float(np.linalg.norm(position - current_position)) > max_translation_m + 1e-9:
        raise ValueError("EEF 恢复目标超过 5 cm")
    old_quaternion = np.asarray(current_eef["quaternion_wxyz"], dtype=float).reshape(-1)
    delta = (Rotation.from_quat(quaternion[[1, 2, 3, 0]])
             * Rotation.from_quat(old_quaternion[[1, 2, 3, 0]]).inv())
    if float(delta.magnitude()) > max_rotation_rad + 1e-9:
        raise ValueError("EEF 恢复目标超过 0.35 rad")


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" or "\U00020000" <= char <= "\U000323af" for char in text)


def validate_public_language(response: Mapping[str, Any]) -> None:
    """公开解释文本禁止 CJK（协议/任务原文不受影响）。"""
    assessment = response.get("assessment", {})
    progress = assessment.get("task_progress", {}) if isinstance(assessment, Mapping) else {}
    values: list[Any] = [response.get("reason")]
    if isinstance(assessment, Mapping):
        values.extend(assessment.get(key) for key in (
            "current_subgoal", "execution_evidence", "expected_next_intent",
            "predicted_next_intent", "intent_evidence"))
    if isinstance(progress, Mapping):
        values.append(progress.get("currently_attempting"))
        values.extend(progress.get("verified_completed", []) or [])
        values.extend(progress.get("remaining", []) or [])
    if any(isinstance(value, str) and _contains_cjk(value) for value in values):
        raise ValueError("公开决策与评估文本必须用英文书写")


def validate_execution_contract(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    gate_policy: str | None = None,
) -> None:
    """校验 `student / edit / eef / stop` 执行契约（GPT-as-Policy `validate_response` 的单臂版）。

    `gate_policy == "failure-or-intent"` 时，额外要求评估块通过
    `gate_assessment.validate_assessment`（若可用）且公开文本为英文。
    """
    if response.get("request_id") != request["request_id"]:
        raise ValueError("响应陈旧或属于另一次观测")
    if not isinstance(response.get("reason"), str) or not response["reason"].strip():
        raise ValueError("必须给出可见的观测/决策解释")
    mode = response.get("mode")
    if mode not in ("student", "edit", "eef", "stop"):
        raise ValueError("mode 必须是 student/edit/eef/stop")
    if gate_policy == "failure-or-intent":
        try:
            from .gate_assessment import validate_assessment  # 可选：迁移后才有
        except ImportError:
            pass
        else:
            validate_assessment(response, request)
        validate_public_language(response)
    if mode == "stop":
        return

    steps = response.get("steps")
    cap = _STEP_CAP[mode]
    if type(steps) is not int or not 1 <= steps <= cap:
        raise ValueError(f"{mode} 模式步数必须是 1..{cap} 的整数")
    if mode == "edit":
        edited_targets(request["student_eef_trajectory"], steps, response["edit"])
    elif mode == "eef":
        validate_eef_target(response["target"], request["current_eef"])
