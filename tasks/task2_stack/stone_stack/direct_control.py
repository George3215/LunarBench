"""单臂版 Direct Astra 工具闭环：observe -> move_tcp -> observe。

参考 GPT-as-Policy 的 robodojo_start/robodojo_act 分工；这里绑定 TASK2 的世界系、
单臂 TCP 和净开口。StoneTools.specs/dispatch 可直接供其他 agent host 注册，
不依赖 Qwen、Codex 或上游的 RoboDojo 服务。
"""
from __future__ import annotations

import json
import jsonschema
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# 直接策略的到位精度；与动作允许的 5 cm / 0.35 rad 上限不同。
TCP_POSITION_TOLERANCE_M = 0.002
TCP_ROTATION_TOLERANCE_RAD = 0.02


def quaternion(matrix):
    """SciPy 默认 xyzw，工具协议统一 wxyz。"""
    return np.roll(Rotation.from_matrix(matrix).as_quat(), 1).tolist()


def tool(name, description, **properties):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False},
    }}


class StoneTools:
    """唯一动作入口。只检查动作契约；不修复输出、不替模型抓取、不回退脚本。"""

    def __init__(self, executor, config):
        self.executor = executor
        self.controller = executor.controller
        self.config = config
        synthetic = executor.synthetic_gripper
        self.max_opening = (2 * synthetic["half_travel"] - executor.pad_thickness_m
                            if synthetic is not None else executor.profile.gripper.max_opening_m)
        self.finished = False
        self.success = False
        self.stable_since = None
        self.reason = ""
        self.failures = []
        self.plans = {}
        self.step_id = 0
        self.last_target = None
        self.repeated_target_count = 0
        self.no_motion_count = 0
        def vector(size):
            return {"type": "array", "items": {"type": "number"}, "minItems": size, "maxItems": size}
        pose = {"type": "object", "properties": {"position": vector(3), "quaternion_wxyz": vector(4)},
                "required": ["position", "quaternion_wxyz"], "additionalProperties": False}
        text = {"type": "string"}
        reason = {"type": "string", "description": "可见依据、动作目的与简短任务进度"}
        self.specs = [
            tool("move_tcp_delta", "Translate by a model-chosen world displacement while holding measured orientation. "
                 "Euclidean displacement must be at most max_delta_m. No automatic waypoint choice.",
                 delta_position=vector(3),
                 grip_width_m={"type": "number", "minimum": 0, "maximum": self.max_opening,
                               "description": "Inner opening in meters: 0 closes, max opens. Smaller means closing."},
                 frame={"type": "string", "enum": ["world"]},
                 steps={"type": "integer", "minimum": 1, "maximum": 5}, reason=reason),
            tool("move_tcp", "Move toward an absolute world TCP pose, then return fresh RGB/proprio. "
                 f"Each call advances at most {config['max_delta_m']} m and {config['max_rotation_rad']} rad toward the target. "
                 "Targets beyond either bound are rejected without execution. "
                 "grip_width_m is the inner opening in meters. Repeat measured pose to only move fingers.",
                 position=vector(3), quaternion_wxyz={"anyOf": [vector(4), {"type": "null"}],
                    "description": "null means hold the current measured orientation; otherwise unit wxyz"},
                 grip_width_m={"type": "number", "minimum": 0, "maximum": self.max_opening,
                               "description": "Inner opening in meters: 0 closes, max opens. Smaller means closing."},
                 frame={"type": "string", "enum": ["world"]},
                 steps={"type": "integer", "minimum": 1, "maximum": 5}, reason=reason),
            tool("plan_stone", "Record your chosen grasp/place TCP poses. Does not move anything.",
                 stone=text, grasp_pose=pose, place_pose=pose, reason=reason),
            tool("record_failure", "Record observed failure, cause hypothesis and concrete next adjustment. No motion.",
                 stone=text, stage=text, evidence=text, cause_hypothesis=text, next_change=text),
            tool("finish", "Request an environment completion check. Does not end an unfinished episode.",
                 reason=reason),
        ]

    def context(self):
        e = self.executor
        g = e.profile.gripper
        pos, rot = self.controller.tcp_pose()
        # 使用装配后实际指垫，兼容场景替换的平行夹爪。
        pad_center = np.mean(e.data.geom_xpos[e._pad_geom_ids[:2]], axis=0)
        pad_offset = rot.T @ (pad_center - pos)
        return dict(arm=e.profile.name, arm_count=1, control_frame="world",
                    control_step_seconds=self.config["servo_seconds"], world_axes="+Z up; right-handed, meters",
                    wall_center=e.workcell.wall_center.tolist(), courses=list(e.workcell.courses),
                    pad_offset_tcp=pad_offset.tolist(), open_axis_tcp=list(g.open_axis_tcp),
                    approach_axis_tcp=list(g.approach_axis_tcp),
                    stone_names=e.stone_names(), stone_count=len(e.stone_names()),
                    max_grip_width_m=self.max_opening,
                    max_delta_m=self.config["max_delta_m"],
                    max_rotation_rad=self.config["max_rotation_rad"])

    def observe(self):
        # 不调用 Executor.observe：它会计算石块真值、候选和抓取规划。
        e = self.executor
        pos, rot = self.controller.tcp_pose()
        frames = e.cameras.capture(tuple(self.config["cameras"]), want_depth=False)
        state = dict(step_id=self.step_id, observation_index=self.step_id,
                     rollout_finished=self.finished, success=self.success, frame="world",
                     termination_reason=("success" if self.success else "step_limit") if self.finished else None,
                     stone_plans=self.plans, recorded_failures=self.failures[-3:],
                     max_steps=self.config["max_steps"], remaining_steps=self.config["max_steps"] - self.step_id,
                     latest_failure=self.failures[-1:], latest_stone_plan=list(self.plans.values())[-1:],
                     time=float(e.data.time), tcp_position=pos.tolist(),
                     tcp_quaternion_wxyz=quaternion(rot),
                     joints=self.controller.arm_qpos().tolist(),
                     grip_width_m=e.measured_opening() - e.pad_thickness_m,
                     cameras={name: dict(position=f.pos.tolist(), rotation=f.rot.tolist(),
                                         fovy_deg=f.fovy_deg, width=f.width, height=f.height)
                              for name, f in frames.items()})
        return state, frames

    def move_tcp(self, position, quaternion_wxyz, grip_width_m, reason, frame, steps):
        pos = np.asarray(position, dtype=float)
        current_pos, current_rot = self.controller.tcp_pose()
        # null 是模型明确选择保持姿态，不是校验失败后的自动修复。
        quat = np.asarray(quaternion(current_rot) if quaternion_wxyz is None else quaternion_wxyz, dtype=float)
        # 外部动作只在此处校验一次。数值/幅度检查属于控制契约，不能由 schema 代替。
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[pos, quat, grip_width_m]).all():
            return dict(status="validation_rejected", executed=False, error="Expected finite xyz, wxyz and grip_width_m")
        if abs(np.linalg.norm(quat) - 1) > 1e-3:
            return dict(status="validation_rejected", executed=False, error="quaternion_wxyz must have unit norm")
        rot = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
        # 校验目标，不截短、不替模型选择中间目标；所有检查都在仿真推进之前。
        distance = np.linalg.norm(pos - current_pos)
        angle = Rotation.from_matrix(rot @ current_rot.T).magnitude()
        if distance > self.config["max_delta_m"] + 1e-10:
            return dict(status="validation_rejected", executed=False, error="TCP translation exceeds max_delta_m",
                        distance_m=float(distance), limit_m=self.config["max_delta_m"])
        if angle > self.config["max_rotation_rad"] + 1e-10:
            return dict(status="validation_rejected", executed=False, error="TCP rotation exceeds max_rotation_rad",
                        angle_rad=float(angle), limit_rad=self.config["max_rotation_rad"])
        if not 0 <= grip_width_m <= self.max_opening:
            return dict(status="validation_rejected", executed=False, error="grip_width_m exceeds gripper opening")
        opening_before = self.executor.measured_opening() - self.executor.pad_thickness_m
        result = self.controller.servo_to(
            pos, rot, seconds=steps * self.config["servo_seconds"],
            preserve_hold_ctrl=True, tolerance_pos=TCP_POSITION_TOLERANCE_M,
            tolerance_rot=TCP_ROTATION_TOLERANCE_RAD,
            grip_ctrl=self.executor._ctrl_for_width(grip_width_m),
            # 即使 TCP 已到位，也要给夹爪完整执行时间。
            hold_chunks=int(np.ceil(steps * self.config["servo_seconds"] / self.executor.model.opt.timestep)) + 1,
            on_step=lambda controller: self.executor._sync_viewer(),
        )
        # 返回实测残差：目标被接受不代表到达或抓取成功。
        final_pos, final_rot = self.controller.tcp_pose()
        metrics = asdict(result)
        metrics.update(
            status="executed", executed=True, control_steps=steps,
            requested_position=pos.tolist(), servo_position=pos.tolist(),
            requested_quaternion_wxyz=quat.tolist(), servo_quaternion_wxyz=quat.tolist(),
            requested_pos_error_m=float(np.linalg.norm(pos - final_pos)),
            requested_rot_error_rad=float(Rotation.from_matrix(rot @ final_rot.T).magnitude()),
        )

        metrics["reached"] = metrics["requested_pos_error_m"] < TCP_POSITION_TOLERANCE_M and metrics["requested_rot_error_rad"] < TCP_ROTATION_TOLERANCE_RAD
        opening_after = self.executor.measured_opening() - self.executor.pad_thickness_m
        target = np.r_[pos, rot.ravel(), grip_width_m]
        repeated = self.last_target is not None and np.allclose(target, self.last_target, atol=1e-4, rtol=0)
        self.repeated_target_count = self.repeated_target_count + 1 if repeated else 1
        self.last_target = target
        moved = float(np.linalg.norm(final_pos - current_pos))
        rotated = float(Rotation.from_matrix(final_rot @ current_rot.T).magnitude())
        gap_change = float(opening_after - opening_before)
        stationary = moved < 0.001 and rotated < 0.01 and abs(gap_change) < 0.001
        self.no_motion_count = self.no_motion_count + 1 if stationary else 0
        metrics.update(measured_translation_m=moved, measured_rotation_rad=rotated,
                       measured_grip_width_m=float(opening_after), grip_width_change_m=gap_change,
                       commanded_grip_width_m=grip_width_m,
                       repeated_target_count=self.repeated_target_count,
                       consecutive_no_motion_calls=self.no_motion_count)
        # 只报告事实，不基于 reason 猜意图，也不擅自替换动作。
        return metrics

    def move_tcp_delta(self, delta_position, grip_width_m, frame, steps, reason):
        """模型给出局部位移并选择保持姿态；只做坐标加法，复用同一门控和执行器。"""
        position, _ = self.controller.tcp_pose()
        return self.move_tcp(position + np.asarray(delta_position), None, grip_width_m, reason, frame, steps)

    def finish(self, reason):
        self.reason = reason
        return dict(status="completion_check_requested", reason=reason, rollout_finished=False)

    def plan_stone(self, stone, grasp_pose, place_pose, reason):
        plan = dict(stone=stone, grasp_pose=grasp_pose, place_pose=place_pose, reason=reason, step_id=self.step_id)
        self.plans[stone] = plan
        return plan

    def record_failure(self, stone, stage, evidence, cause_hypothesis, next_change):
        failure = dict(step_id=self.step_id, stone=stone, stage=stage, evidence=evidence,
                       cause_hypothesis=cause_hypothesis, next_change=next_change)
        self.failures.append(failure)
        return failure

    def dispatch(self, name, arguments):
        specs = {t["function"]["name"]: t["function"]["parameters"] for t in self.specs}
        if name not in specs:
            return dict(status="validation_rejected", executed=False, error="Unknown tool")
        try:
            jsonschema.validate(arguments, specs[name])
        except jsonschema.ValidationError as error:
            return dict(status="validation_rejected", executed=False, error=error.message)
        return {"move_tcp": self.move_tcp, "move_tcp_delta": self.move_tcp_delta, "finish": self.finish,
                "plan_stone": self.plan_stone, "record_failure": self.record_failure}[name](**arguments)

    def check_terminal(self):
        """Python 私有评估：合格观测跨越 1.2 秒才成功；两次采样之间不作连续稳定保证。"""
        from .evaluation import evaluate_wall
        evaluation = evaluate_wall(self.executor)
        now = float(self.executor.data.time)
        if not evaluation["success"]:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = now
        self.success = self.stable_since is not None and now - self.stable_since >= 1.2
        self.finished = self.success or self.step_id >= self.config["max_steps"]
        return evaluation



def run_direct(executor, policy, instruction: str, output: Path, lessons=None, initialized=False) -> dict:
    """一次初始归位；随后每次工具调用都记录执行前后观测。异常保留已写入轨迹。"""
    output.mkdir(parents=True, exist_ok=False)
    e = executor
    if not initialized:
        e.controller.set_arm_qpos(np.asarray(e.profile.home_qpos))
        e.controller.forward()
        e.controller.set_arm_ctrl(e.controller.arm_qpos())
        e.controller.set_gripper_ctrl(e.profile.gripper.open_ctrl if e.synthetic_gripper is None
                                     else e._ctrl_for_width(e.profile.gripper.max_opening_m))
        e.settle(policy.config["initial_settle_s"])
    tools = StoneTools(e, policy.config)
    context = tools.context()
    context["previous_trial_lessons"] = lessons or {}
    policy.reset(instruction, context, tools.specs)
    (output / "context.json").write_text(json.dumps(dict(instruction=instruction, **context), indent=2))

    def capture(index):
        state, frames = tools.observe()
        from PIL import Image
        for name, frame in frames.items():
            Image.fromarray(frame.rgb).save(output / f"{index:04d}_{name}.jpg")
        (output / f"{index:04d}_state.json").write_text(json.dumps(state, indent=2))
        # 仿真状态仅存证，不传给 Qwen；用于离线定位滑落/接触问题。
        np.savez_compressed(output / f"{index:04d}_sim.npz", time=e.data.time,
                            qpos=e.data.qpos, qvel=e.data.qvel, ctrl=e.data.ctrl)
        return state, frames

    state, frames = capture(0)
    count = 0
    with (output / "trajectory.jsonl").open("w", buffering=1) as trace:
        for index in range(policy.config["max_steps"]):
            call = policy.decide(state, frames)
            name, arguments = call["name"], call["arguments"]
            # 策略层已经完成唯一一次解码；执行器只接收结构化参数。
            trace.write(json.dumps(dict(event="call", step=index, call=call, usage=policy.last_usage, policy_observation=policy.last_observation), ensure_ascii=False) + "\n")
            e._log(f"[direct {index}] {name} {arguments}")
            result = tools.dispatch(name, arguments)
            tools.step_id = index + 1
            evaluation = tools.check_terminal()
            result.update(rollout_finished=tools.finished, success=tools.success)
            e._log(f"[direct {index} result] {result}")
            state, frames = capture(index + 1)
            trace.write(json.dumps(dict(event="result", step=index, result=result, state=state), ensure_ascii=False) + "\n")
            (output / "failures.json").write_text(json.dumps(tools.failures, ensure_ascii=False, indent=2))
            (output / "stone_plans.json").write_text(json.dumps(tools.plans, ensure_ascii=False, indent=2))
            policy.record_result(call, result)
            count += name in ("move_tcp", "move_tcp_delta") and result.get("executed", False)
            if tools.finished:
                break
    return dict(arm=e.profile.name, policy=policy.name,
                tool_calls=tools.step_id, failures=tools.failures, stone_plans=tools.plans,
                status="success" if tools.success else "step_limit",
                evaluation=dict(evaluation, success=tools.success, stable_observation_seconds=1.2),
                success=tools.success, reason=tools.reason, executed_actions=count,
                artifact_dir=str(output.resolve()))
