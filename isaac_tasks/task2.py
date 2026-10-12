"""独立 Isaac Lab TASK2：USD + manifest → GPU 物理、局部 EEF 控制与课程奖励。

本模块只依赖 Isaac Lab / PyTorch，不导入 MuJoCo，也不读取原 task2 源代码。
所有位置以各环境原点为参考；并行环境只添加 scene.env_origins 平移。
"""

import json
from pathlib import Path

import torch
from isaaclab import sim
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    TiledCamera,
    TiledCameraCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
)
from pxr import PhysxSchema, UsdPhysics

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "assets/task2/manifest.json").read_text())
PREFIX = "/World/envs/env_.*/Scene"


@configclass
class Task2Cfg(DirectRLEnvCfg):
    seed = 42
    # reset 写入关节状态后更新 Fabric/RGB，保证腕部相机首帧对应新位姿。
    num_rerenders_on_reset = 1
    viewer = ViewerCfg(
        eye=(1.4, -1.4, 1.2), lookat=(0.0, 0.0, 0.15), origin_type="env", env_index=0
    )
    decimation = 160
    episode_length_s = 240.0
    action_space = 7
    observation_space = 185
    state_space = 0
    sim = sim.SimulationCfg(
        dt=0.0015,
        render_interval=160,
        gravity=tuple(META["gravity"]),
        physx=sim.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=16,
            min_velocity_iteration_count=4,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**18,
        ),
    )
    scene = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)
    stage = "grasp"
    cameras = False
    observation_mode = "state"
    camera_width = 640
    camera_height = 480
    reward_weights = META["reward"]
    curriculum = META["curriculum"]
    max_delta_m = 0.02
    max_rotation_rad = 0.1


class Task2Env(DirectRLEnv):
    def _setup_scene(self):
        cfg = sim.UsdFileCfg(usd_path=str(ROOT / "assets/task2/scene.usda"))
        cfg.func("/World/envs/env_0/Scene", cfg)
        stage = sim.get_current_stage()
        PhysxSchema.PhysxArticulationAPI.Apply(
            stage.GetPrimAtPath("/World/envs/env_0/Scene/Robot")
        ).CreateEnabledSelfCollisionsAttr(True)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr(0.001)
                collision.CreateRestOffsetAttr(0.0)
        for item in META["mimic"]:
            root = "/World/envs/env_0/Scene/Robot/Joints/"
            joint = stage.GetPrimAtPath(root + item["joint"])
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(joint, "rotX")
            mimic.CreateReferenceJointRel().SetTargets([root + item["reference"]])
            mimic.CreateReferenceJointAxisAttr("rotX")
            mimic.CreateGearingAttr(item["gearing"])
            mimic.CreateOffsetAttr(0.0)
        # 同参数关节共用 actuator，避免每个物理步重复发射六组小 CUDA 运算。
        groups = {}
        for actuator in META["actuators"]:
            key = (
                actuator["kp"],
                actuator["kd"],
                max(abs(v) for v in actuator["force_range"]) or 300.0,
            )
            groups.setdefault(key, []).append(actuator["joint"])
        actuators = {}
        for index, ((kp, kd, limit), joints) in enumerate(groups.items()):
            actuators[str(index)] = ImplicitActuatorCfg(
                joint_names_expr=joints,
                stiffness=kp,
                damping=kd,
                effort_limit_sim=limit,
            )
        self.robot = Articulation(
            ArticulationCfg(
                prim_path=PREFIX + "/Robot",
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(
                    joint_pos={n: v["position"] for n, v in META["joints"].items()}
                ),
                actuators=actuators,
            )
        )
        self.stones = []
        for stone in META["stones"]:
            obj = RigidObject(
                RigidObjectCfg(
                    prim_path=PREFIX + "/Stones/" + stone["name"],
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(stone["position"]),
                        rot=tuple(stone["quaternion_wxyz"]),
                    ),
                )
            )
            self.stones.append(obj)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        for i, obj in enumerate(self.stones):
            self.scene.rigid_objects[f"stone_{i}"] = obj
        # 每块石头的 13 列接触力：10 块石头、地面、左右手指刚体。
        stone_paths = [PREFIX + "/Stones/" + s["name"] for s in META["stones"]]
        self.contacts = []
        for i, path in enumerate(stone_paths):
            filters = (
                stone_paths
                + [PREFIX + "/Static/geom_0"]
                + [PREFIX + "/Robot/" + p["body"] for p in META["pads"]]
            )
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=path,
                    filter_prim_paths_expr=filters,
                    update_period=0.0,
                    history_length=0,
                    max_contact_data_count_per_prim=128,
                )
            )
            self.contacts.append(sensor)
            self.scene.sensors[f"contact_{i}"] = sensor
        self.cameras = {}
        if self.cfg.cameras:
            for name, c in META["cameras"].items():
                path = c["path"].replace("/Scene", PREFIX, 1)
                cam = TiledCamera(
                    TiledCameraCfg(
                        prim_path=path,
                        spawn=None,
                        width=self.cfg.camera_width,
                        height=self.cfg.camera_height,
                        data_types=["rgb"],
                        update_latest_camera_pose=True,
                        update_period=0.0,
                    )
                )
                self.cameras[name] = cam
                self.scene.sensors[name] = cam
        sim.DomeLightCfg(intensity=450.0).func(
            "/World/Light", sim.DomeLightCfg(intensity=450.0)
        )

    def __init__(self, cfg, **kwargs):
        if not (0 < cfg.max_delta_m <= 0.05 and 0 < cfg.max_rotation_rad <= 0.35):
            raise ValueError("EEF limits must be within 0.05 m and 0.35 rad")
        if cfg.observation_mode == "visual" and not cfg.cameras:
            raise ValueError("Visual policy requires cameras=true")
        super().__init__(cfg, **kwargs)
        self.arm_ids = self.robot.find_joints(META["arm_joints"], preserve_order=True)[
            0
        ]
        self.finger_ids = self.robot.find_joints(
            ["finger_joint", "right_outer_knuckle_joint"], preserve_order=True
        )[0]
        self.tcp_id = self.robot.find_bodies(META["tcp"]["body"])[0][0]
        self.tcp_offset = torch.tensor(
            META["tcp"]["position"], device=self.device
        ).expand(self.num_envs, -1)
        self.tcp_rotation = torch.tensor(
            META["tcp"]["quaternion_wxyz"], device=self.device
        ).expand(self.num_envs, -1)
        self.pad_ids = self.robot.find_bodies(
            [p["body"] for p in META["pads"]], preserve_order=True
        )[0]
        self.pad_offsets = torch.tensor(
            [p["position"] for p in META["pads"]], device=self.device
        )
        self.pad_rotation = torch.tensor(
            [p["quaternion_wxyz"] for p in META["pads"]], device=self.device
        )
        self.pad_size = torch.tensor(
            [p["size"] for p in META["pads"]], device=self.device
        )
        self.center_offsets = torch.tensor(
            [s["center_offset"] for s in META["stones"]], device=self.device
        )
        self.targets = torch.tensor(META["targets"], device=self.device)
        self.initial_z = torch.tensor(
            [s["initial_center_z"] for s in META["stones"]], device=self.device
        )
        self.actions = torch.zeros((self.num_envs, 7), device=self.device)
        self.q_target = self.robot.data.default_joint_pos.clone()
        self.grasp_time = torch.zeros((self.num_envs, 10), device=self.device)
        self.lift_time = torch.zeros_like(self.grasp_time)
        self.place_time = torch.zeros_like(self.grasp_time)
        self.credited = torch.zeros(
            (self.num_envs, 10, 3), device=self.device, dtype=torch.bool
        )
        self.previous_relative = torch.zeros((self.num_envs, 10, 3), device=self.device)
        self.previous_phi = torch.full((self.num_envs,), torch.nan, device=self.device)
        self.wall_time = torch.zeros_like(self.previous_phi)
        self._control_tick = 0
        self.target_pos = self.tcp_pose()[0].clone()
        self.target_quat = self.tcp_pose()[1].clone()

    def tcp_pose(self):
        rotation = self.robot.data.body_quat_w[:, self.tcp_id]
        position = self.robot.data.body_pos_w[:, self.tcp_id] + quat_apply(
            rotation, self.tcp_offset
        )
        return position, quat_mul(rotation, self.tcp_rotation)

    def _pre_physics_step(self, actions):
        # 初始化势函数和相对位姿，不给 reset 本身发奖励。
        centers = self.stone_centers()
        pads = self.pad_positions() - self.scene.env_origins[:, None]
        reach = torch.exp(
            -torch.linalg.vector_norm(centers - pads.mean(1)[:, None], dim=-1) / 0.15
        )
        distances = torch.linalg.vector_norm(
            centers[:, :, None] - pads[:, None], dim=-1
        )
        centered = reach * torch.exp(
            -(distances[:, :, 0] - distances[:, :, 1]).abs() / 0.025
        )
        limit = 10 if self.cfg.stage in ("grasp", "lift") else 4
        initial_phi = (
            self.cfg.reward_weights["reach"] * reach[:, :limit].max(-1).values
            + self.cfg.reward_weights["center"] * centered[:, :limit].max(-1).values
        )
        fresh = torch.isnan(self.previous_phi)
        self.previous_phi = torch.where(fresh, initial_phi, self.previous_phi)
        tcp, rotation = self.tcp_pose()
        relative = quat_apply(
            quat_inv(rotation)[:, None].expand(-1, 10, -1),
            centers + self.scene.env_origins[:, None] - tcp[:, None],
        )
        self.previous_relative = torch.where(
            fresh[:, None, None], relative, self.previous_relative
        )
        self.actions = actions.clamp(-1, 1)
        dp = self.actions[:, :3]
        dp = (
            dp
            / torch.linalg.vector_norm(dp, dim=-1, keepdim=True).clamp_min(1.0)
            * self.cfg.max_delta_m
        )
        dr = self.actions[:, 3:6]
        dr = (
            dr
            / torch.linalg.vector_norm(dr, dim=-1, keepdim=True).clamp_min(1.0)
            * self.cfg.max_rotation_rad
        )
        self.target_pos = self.tcp_pose()[0] + dp
        angle = torch.linalg.vector_norm(dr, dim=-1)
        dq = quat_from_angle_axis(angle, dr / angle[:, None].clamp_min(1e-8))
        self.target_quat = quat_mul(dq, self.tcp_pose()[1])
        self._control_tick = 0

    def _apply_action(self):
        # 小步 DLS：只在当前实测构型附近求解，不规划、不切换 IK 分支。
        if self._control_tick % 8 == 0:
            pos = self.tcp_pose()[0]
            rot = self.tcp_pose()[1]
            dp = self.target_pos - pos
            dp = dp * (
                0.01
                / torch.linalg.vector_norm(dp, dim=-1, keepdim=True).clamp_min(0.01)
            )
            dr = axis_angle_from_quat(quat_mul(self.target_quat, quat_inv(rot)))
            dr = dr * (
                0.05
                / torch.linalg.vector_norm(dr, dim=-1, keepdim=True).clamp_min(0.05)
            )
            jac = self.robot.root_physx_view.get_jacobians()[:, self.tcp_id - 1, :, :][
                :, :, self.arm_ids
            ]
            offset = quat_apply(
                self.robot.data.body_quat_w[:, self.tcp_id], self.tcp_offset
            )
            jac = jac.clone()
            jac[:, :3] += torch.cross(
                jac[:, 3:].transpose(1, 2), offset[:, None].expand(-1, 6, -1), dim=-1
            ).transpose(1, 2)
            error = torch.cat([dp, dr], dim=-1)
            dq = (
                jac.transpose(1, 2)
                @ torch.linalg.solve(
                    jac @ jac.transpose(1, 2)
                    + 0.03**2 * torch.eye(6, device=self.device),
                    error.unsqueeze(-1),
                )
            ).squeeze(-1)
            q = self.robot.data.joint_pos[:, self.arm_ids]
            # PhysX 用重力前馈支撑机械臂；零末端动作不应不断下沉。
            goal = q + dq.clamp(-0.03, 0.03)
            self.robot.set_joint_effort_target(
                self.arm_feedforward(jac), joint_ids=self.arm_ids
            )
            limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
            self.q_target[:, self.arm_ids] = goal.clamp(
                limits[:, :, 0], limits[:, :, 1]
            )
            gripper = META["gripper"]
            width = (self.actions[:, 6] + 1) / 2 * gripper["max_opening_m"]
            value = (
                width + gripper["pad_thickness_m"] - gripper["width_intercept"]
            ) / gripper["width_slope"]
            limits = sorted([gripper["open_ctrl"][0], gripper["close_ctrl"][0]])
            value = value.clamp(*limits)
            self.q_target[:, self.finger_ids] = torch.stack([value, -value], dim=1)
            self.robot.set_joint_position_target(self.q_target)
        self._control_tick += 1

    def arm_feedforward(self, jacobian):
        """本体重力前馈；任务控制器可在真实抓住物体后补偿已知载荷。"""
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()
        return gravity[:, self.arm_ids]

    def stone_centers(self):
        positions = torch.stack([s.data.root_pos_w for s in self.stones], dim=1)
        rotations = torch.stack([s.data.root_quat_w for s in self.stones], dim=1)
        return (
            positions
            + quat_apply(rotations, self.center_offsets.expand(self.num_envs, -1, -1))
            - self.scene.env_origins[:, None]
        )

    def pad_positions(self):
        return self.robot.data.body_pos_w[:, self.pad_ids] + quat_apply(
            self.robot.data.body_quat_w[:, self.pad_ids],
            self.pad_offsets.expand(self.num_envs, -1, -1),
        )

    def pad_contacts(self, pads):
        # PhysX 的过滤对象是刚体，进一步以实际接触点落在指垫 box 内过滤指尖碰撞。
        rotations = quat_mul(
            self.robot.data.body_quat_w[:, self.pad_ids],
            self.pad_rotation.expand(self.num_envs, -1, -1),
        )
        magnitudes = []
        for sensor in self.contacts:
            force, point, _normal, distance, count, start = (
                sensor.contact_physx_view.get_contact_data(dt=self.physics_dt)
            )
            count = count.reshape(self.num_envs, -1)[:, -2:]
            start = start.reshape(self.num_envs, -1)[:, -2:]
            slots = torch.arange(128, device=self.device)
            index = (start[:, :, None] + slots).long().clamp(0, len(point) - 1)
            local = quat_apply(
                quat_inv(rotations)[:, :, None].expand(-1, -1, 128, -1),
                point[index] - pads[:, :, None],
            )
            on_pad = (local.abs() <= self.pad_size[None, :, None] + 0.0015).all(-1)
            valid = (
                (slots < count[:, :, None])
                & on_pad
                & (distance[index].squeeze(-1) <= 0.001)
            )
            # PhysX 的接触标量可带负号；阈值比较的是力的大小，不能把负值当无接触。
            magnitudes.append(
                torch.where(valid, force[index].squeeze(-1).abs(), 0.0).sum(-1)
            )
        return torch.stack(magnitudes, dim=1)

    def _signals(self):
        c = self.cfg.curriculum
        r = self.cfg.reward_weights
        pos = self.stone_centers()
        vel = torch.stack([s.data.root_vel_w for s in self.stones], dim=1)
        forces = torch.stack(
            [s.data.force_matrix_w[:, 0] for s in self.contacts], dim=1
        )
        touch = torch.linalg.vector_norm(forces, dim=-1) > c["contact_force_n"]
        resting = touch[:, :, :11].any(-1)
        pads = self.pad_positions()
        pad_touch = self.pad_contacts(pads) >= self.cfg.curriculum["contact_force_n"]
        bilateral = pad_touch.all(-1)
        center = pads.mean(1) - self.scene.env_origins
        tcp = self.tcp_pose()[0] - self.scene.env_origins
        quat = self.tcp_pose()[1]
        relative = quat_apply(
            quat_inv(quat)[:, None].expand(-1, 10, -1), pos - tcp[:, None]
        )
        stable = (
            torch.linalg.vector_norm(relative - self.previous_relative, dim=-1)
            <= c["relative_tolerance_m"]
        )
        height = pos[:, :, 2] - self.initial_z
        retained = bilateral & ~resting & stable
        self.grasp_time = torch.where(
            retained & (height >= c["grasp_height_m"]),
            self.grasp_time + self.step_dt,
            0.0,
        )
        self.lift_time = torch.where(
            retained & (height >= c["lift_height_m"]),
            self.lift_time + self.step_dt,
            0.0,
        )
        self.previous_relative = relative.clone()
        # 支撑边由实际接触力、竖直分量和高度关系共同确定。
        norm = torch.linalg.vector_norm(forces[:, :, :10], dim=-1)
        lower = pos[:, :, 2, None] > pos[:, None, :, 2] + 0.01
        support = touch[:, :, :10] & lower & (forces[:, :, :10, 2].abs() > 0.3 * norm)
        layers = torch.where(touch[:, :, 10], 0, -1)
        for _ in range(4):
            candidate = layers[:, None, :].expand(-1, 10, -1)
            available = support & (candidate >= 0)
            new = (
                available.any(-1)
                & ((~support) | (candidate >= 0)).all(-1)
                & (layers < 0)
            )
            highest = torch.where(support, candidate, -1).max(-1).values + 1
            layers = torch.where(new, highest, layers)
        expected = torch.tensor([0] * 4 + [1] * 3 + [2] * 2 + [3], device=self.device)
        bridges = (support & (layers[:, None, :] == layers[:, :, None] - 1)).sum(
            -1
        ) >= 2
        layer_ok = (layers == expected) & ((expected == 0) | bridges)
        slow = (torch.linalg.vector_norm(vel[:, :, :3], dim=-1) <= 0.015) & (
            torch.linalg.vector_norm(vel[:, :, 3:], dim=-1) <= 0.15
        )
        released = ~touch[:, :, 11:13].any(-1)
        distance = torch.linalg.vector_norm(pos - self.targets, dim=-1)
        close = torch.exp(-distance / 0.1)
        placed = layer_ok & slow & released & (distance < 0.04) & self.credited[:, :, 0]
        self.place_time = torch.where(placed, self.place_time + self.step_dt, 0.0)
        achieved = torch.stack(
            [
                self.grasp_time >= c["grasp_hold_seconds"] - 1e-7,
                self.lift_time >= c["lift_hold_seconds"] - 1e-7,
                self.place_time >= c["place_hold_seconds"] - 1e-7,
            ],
            dim=-1,
        )
        allowed = torch.tensor(
            [1, self.cfg.stage != "grasp", self.cfg.stage not in ("grasp", "lift")],
            device=self.device,
            dtype=torch.bool,
        )
        fresh = achieved & ~self.credited & allowed
        bonus = (
            fresh
            * torch.tensor(
                [c["grasp_bonus"], c["lift_bonus"], c["place_bonus"]],
                device=self.device,
            )
        ).sum((1, 2))
        self.credited |= achieved & allowed
        stage = self.cfg.stage
        required = {
            "grasp": 0,
            "lift": 0,
            "place": 1,
            "stack4": 4,
            "stack7": 7,
            "stack9": 9,
            "stack10": 10,
        }[stage]
        active = ~placed
        if required:
            limit = 4 if stage == "place" else required
            active[:, limit:] = False
            for end in [4, 7, 9]:
                active[:, end:] &= placed[:, : min(end, limit)].all(-1)[:, None]
        reach = torch.exp(
            -torch.linalg.vector_norm(pos - center[:, None], dim=-1) / 0.15
        )
        pd = torch.linalg.vector_norm(
            pos[:, :, None] - (pads - self.scene.env_origins[:, None])[:, None], dim=-1
        )
        centered = reach * torch.exp(-(pd[:, :, 0] - pd[:, :, 1]).abs() / 0.025)
        lift = (height / c["lift_height_m"]).clamp(0, 1) * retained
        terms = {
            "reach": (reach * active).max(-1).values,
            "center": (centered * active).max(-1).values,
            "touch": (pad_touch.float().mean(-1) * active).max(-1).values,
            "grasp": (bilateral * active).max(-1).values,
            "hold": ((self.grasp_time / c["grasp_hold_seconds"]).clamp(0, 1) * active)
            .max(-1)
            .values,
            "lift": (lift * active).max(-1).values,
            "transport": (lift * close * active).max(-1).values * bool(required),
            "place": (close * layer_ok * slow * released * self.credited[:, :, 0])[
                :, : 4 if stage == "place" else required
            ].sum(-1),
        }
        phi = sum(self.cfg.reward_weights[k] * v for k, v in terms.items())
        counts = torch.stack([(layers == i).sum(-1) for i in range(4)], dim=-1)
        wall = (
            (counts == torch.tensor([4, 3, 2, 1], device=self.device)).all(-1)
            & (layers >= 0).all(-1)
            & ((layers == 0) | bridges).all(-1)
            & slow.all(-1)
            & released.all(-1)
            & (pos[:, :, 0].abs() <= 0.45).all(-1)
            & (pos[:, :, 1].abs() <= 0.16).all(-1)
        )
        self.wall_time = torch.where(wall, self.wall_time + self.step_dt, 0.0)
        native = self.wall_time >= 1.2
        if stage == "grasp":
            success = achieved[:, :, 0].any(-1)
        elif stage == "lift":
            success = achieved[:, :, 1].any(-1)
        elif stage == "place":
            success = achieved[:, :4, 2].any(-1)
        elif stage == "stack10":
            success = native
        else:
            success = achieved[:, :required, 2].all(-1)
        fallen = (pos[:, :, 2] < -0.1).any(-1)
        # 残差状态机复用同一份真实接触/成功判据，不另写一套近似成功。
        self.signals = {
            "bilateral": bilateral,
            "pad_touch": pad_touch,
            "stable": stable,
            "achieved": achieved,
            "layer_ok": layer_ok,
            "slow": slow,
            "released": released,
            "placed": placed,
            "distance": distance,
            "fallen": fallen,
            "bonus": bonus,
        }
        success = (success | native) & ~fallen
        done = success | fallen
        reward = (
            0.99 * torch.where(done, 0.0, phi)
            - self.previous_phi
            - r["step_cost"]
            - r["action_cost"] * (self.actions**2).mean(-1)
            + bonus
            + torch.where(native, r["success_bonus"], c["stage_bonus"]) * success
            - r["fall_penalty"] * fallen
        )
        self.previous_phi = phi
        self.extras["log"] = {
            "reward/mean": reward.mean(),
            "success/curriculum_stage": success.float().mean(),
            "success/full_wall": native.float().mean(),
            "grasp/contacts": bilateral.float().sum(-1).mean(),
        }
        return reward, done, success, native

    def _get_dones(self):
        self._result = self._signals()
        return self._result[1], self.episode_length_buf >= self.max_episode_length

    def _get_rewards(self):
        return self._result[0]

    def _get_observations(self):
        pos = self.tcp_pose()[0] - self.scene.env_origins
        from isaaclab.utils.math import matrix_from_quat

        rotation = matrix_from_quat(self.tcp_pose()[1]).reshape(self.num_envs, 9)
        # 特权状态 PPO；相机存在但未偷偷拼入状态策略，像素训练可独立接入。
        values = [
            self.robot.data.joint_pos[:, self.arm_ids],
            self.robot.data.joint_vel[:, self.arm_ids],
            pos,
            rotation,
            (
                torch.linalg.vector_norm(
                    self.pad_positions()[:, 0] - self.pad_positions()[:, 1], dim=-1
                )
                - META["gripper"]["pad_thickness_m"]
            )[:, None],
            self.targets.reshape(1, -1).expand(self.num_envs, -1),
        ]
        for stone in self.stones:
            values.extend(
                [
                    stone.data.root_pos_w - self.scene.env_origins,
                    stone.data.root_quat_w,
                    stone.data.root_vel_w,
                ]
            )
        state = torch.cat(values, dim=-1)
        if self.cfg.observation_mode == "visual":
            # 前 25 维只有关节、TCP、夹爪；石头真值与目标坐标只给 Critic。
            images = torch.stack(
                [
                    camera.data.output["rgb"][..., :3]
                    for camera in self.cameras.values()
                ],
                dim=1,
            )
            return {"policy": state[:, :25], "critic": state, "rgb": images}
        return {"policy": state}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        q = self.robot.data.default_joint_pos[env_ids].clone()
        v = torch.zeros_like(q)
        self.robot.write_joint_state_to_sim(q, v, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        for obj in self.stones:
            state = obj.data.default_root_state[env_ids].clone()
            state[:, :3] += self.scene.env_origins[env_ids]
            obj.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
            obj.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
        self.grasp_time[env_ids] = 0.0
        self.lift_time[env_ids] = 0.0
        self.place_time[env_ids] = 0.0
        self.wall_time[env_ids] = 0.0
        self.credited[env_ids] = False
        self.previous_phi[env_ids] = torch.nan
        self.previous_relative[env_ids] = 0.0
        self.q_target[env_ids] = q
        self.actions[env_ids] = 0.0
