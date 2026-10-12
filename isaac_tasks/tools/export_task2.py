"""TASK2 离线适配器；导出以后 Isaac Lab 训练无需 MuJoCo 或 task2 源码。"""

import json
import logging
import sys
from pathlib import Path

import mujoco
from mujoco_usd import export_model
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "tasks/task2_stack"
sys.path.insert(0, str(SOURCE))
from stone_stack.rl.env import StoneStackEnv
from stone_stack.task_config import load_config


def export():
    cfg = load_config(SOURCE / "configs/rl_curriculum_01_grasp.yaml")
    cfg["policy"]["rl"]["observation"] = "state"
    env = StoneStackEnv(cfg)
    m = env.model
    initial = env.data
    names = env.names
    output = ROOT / "assets/task2"
    output.mkdir(parents=True, exist_ok=True)
    meta, owners, rest = export_model(
        m, initial, output, names, cfg["policy"]["rl"]["cameras"]
    )
    site = m.site(env.profile.tcp_site).id
    tcp_body = int(m.site_bodyid[site])
    tcp_owner = owners[tcp_body]
    tcp_rotation = rest.xmat[tcp_owner].reshape(3, 3)
    tcp_position = tcp_rotation.T @ (rest.site_xpos[site] - rest.xpos[tcp_owner])
    tcp_quat = Rotation.from_matrix(
        tcp_rotation.T @ rest.site_xmat[site].reshape(3, 3)
    ).as_quat()[[3, 0, 1, 2]]
    meta.update(
        version=1,
        source_mujoco=mujoco.__version__,
        timestep=float(m.opt.timestep),
        gravity=m.opt.gravity.tolist(),
        arm_joints=list(env.profile.joints),
        gripper={
            "max_opening_m": env.profile.gripper.max_opening_m,
            "pad_thickness_m": env.e.pad_thickness_m,
            "width_intercept": env.profile.gripper.width_intercept,
            "width_slope": env.profile.gripper.width_slope,
            "open_ctrl": list(env.profile.gripper.open_ctrl),
            "close_ctrl": list(env.profile.gripper.close_ctrl),
        },
        tcp={
            "body": m.body(tcp_owner).name,
            "position": tcp_position.tolist(),
            "quaternion_wxyz": tcp_quat.tolist(),
        },
        pads=[
            {
                "body": m.body(m.geom_bodyid[g]).name,
                "geom": g,
                "position": m.geom_pos[g].tolist(),
                "quaternion_wxyz": m.geom_quat[g].tolist(),
                "size": m.geom_size[g].tolist(),
            }
            for g in env.e._pad_geom_ids[:2]
        ],
        stones=[
            {
                "name": n,
                "position": initial.xpos[m.body(n).id].tolist(),
                "quaternion_wxyz": initial.xquat[m.body(n).id].tolist(),
                "mass": float(m.body_mass[m.body(n).id]),
                "initial_center_z": float(env.initial_z[k]),
                "center_offset": (
                    initial.xmat[m.body(n).id].reshape(3, 3).T
                    @ (env.e.stone_center(n) - initial.xpos[m.body(n).id])
                ).tolist(),
            }
            for k, n in enumerate(names)
        ],
        initial_tcp_position=env.e.controller.tcp_pose()[0].tolist(),
        targets=env.targets.tolist(),
        actuators=[
            {
                "joint": m.joint(int(m.actuator_trnid[i, 0])).name,
                "kp": float(m.actuator_gainprm[i, 0]),
                "kd": float(-m.actuator_biasprm[i, 2]),
                "force_range": m.actuator_forcerange[i].tolist(),
            }
            for i in range(m.nu)
        ],
        curriculum=cfg["policy"]["rl"]["curriculum"],
        reward=cfg["policy"]["rl"]["reward"],
    )
    (output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    env.close()
    logger.info(
        "Exported %s: %d bodies, %d geoms, %d coupling constraints",
        output,
        m.nbody,
        m.ngeom,
        len(meta["mimic"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export()
