"""MuJoCo simulation and UE-display USD checks for the arms in assets/robots/.

For each arm directory: load `arm.xml`, confirm the home keyframe is
self-consistent, step it under a commanded motion and require a finite state and
real tool-tip displacement, and open the `.usdz` with usd-core and compare every
link frame and every mesh placement against MuJoCo's own kinematics. The USD is
authored at qpos0, so that is the pose it is compared at.

Needs `usd-core`. Writes `.local/validation/arms.json`.
"""
from pathlib import Path
import json
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ROOT / 'assets/robots'
NAMES = ('ur10', 'ur10e', 'ur12e', 'so101')
OUT = ROOT.parent / '.local/validation/arms.json'


def check_mujoco(arm):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ARMS / arm / 'arm.xml'))
    result = {'nq': model.nq, 'nu': model.nu, 'nbody': model.nbody,
              'bodies': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                         for i in range(1, model.nbody)],
              'joints': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                         for i in range(model.njnt)]}
    assert model.nq == 6 and model.nu == 6, (arm, model.nq, model.nu)
    assert model.site('tool_tip').id >= 0
    assert model.nkey == 1 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, 0) == 'home'

    # The home keyframe must not be standing in self-penetration: contacts there
    # would load the joints with forces no command asked for.
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    result['home_contacts'] = int(data.ncon)
    assert data.ncon == 0, (arm, data.ncon)

    # Held at home with zero control error the arm must only sag under gravity.
    # Without gravity it must hold the keyframe exactly, which is what shows the
    # residual is weight and not a solver or contact artifact.
    for _ in range(1000):
        mujoco.mj_step(model, data)
    result['home_droop_rad'] = float(np.abs(data.qpos - model.key_qpos[0]).max())
    gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0
    level = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, level, 0)
    for _ in range(1000):
        mujoco.mj_step(model, level)
    result['home_droop_without_gravity_rad'] = float(np.abs(level.qpos - model.key_qpos[0]).max())
    model.opt.gravity[:] = gravity
    assert result['home_droop_without_gravity_rad'] < 1e-9, arm
    assert result['home_droop_rad'] < 0.1, (arm, result['home_droop_rad'])

    # Command a joint away from home and require the tip to follow.
    tip = model.site('tool_tip').id
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    start = data.site_xpos[tip].copy()
    target = data.ctrl.copy()
    low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    target[1] = float(np.clip(target[1] + 0.3, low[1], high[1]))
    for _ in range(2000):
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qacc).all()
    result['tool_tip_displacement_m'] = float(np.linalg.norm(data.site_xpos[tip] - start))
    result['joint_tracking_error_rad'] = float(abs(data.qpos[1] - target[1]))
    assert result['tool_tip_displacement_m'] > 0.05, (arm, result['tool_tip_displacement_m'])

    # A servo tuned stiffer than its torque limit allows does not settle: it
    # chatters against the limit while the joint barely moves, which a short run
    # reads as success. Keep stepping, then require the arm to come to rest well
    # inside its force limits.
    for _ in range(6000):
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
    result['settled_qvel_max'] = float(np.abs(data.qvel).max())
    result['force_headroom'] = float((np.abs(data.actuator_force)
                                      / np.abs(model.actuator_forcerange[:, 1])).max())
    assert result['settled_qvel_max'] < 1e-3, (arm, result['settled_qvel_max'])
    assert result['force_headroom'] < 0.9, (arm, result['force_headroom'])

    # Replaying a saved state must reproduce it bit for bit, which is what a
    # task-side rollout depends on.
    saved = data.qpos.copy()
    expected = data.qvel.copy()
    for _ in range(3):
        mujoco.mj_step(model, data)
    data.qpos[:], data.qvel[:] = saved, expected
    for _ in range(3):
        mujoco.mj_step(model, data)
    replay = np.concatenate([data.qpos, data.qvel])
    data.qpos[:], data.qvel[:] = saved, expected
    for _ in range(3):
        mujoco.mj_step(model, data)
    result['replay_error'] = float(np.abs(replay - np.concatenate([data.qpos, data.qvel])).max())
    assert result['replay_error'] == 0.0, (arm, result['replay_error'])
    return model, result


def check_usd(arm, model):
    from pxr import Usd, UsdGeom
    import mujoco
    destination = ARMS / arm / f'{arm}.usdz'
    with tempfile.TemporaryDirectory() as scratch:
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(scratch)
        stage = Usd.Stage.Open(str(Path(scratch) / f'{arm}.usd'))
        assert stage, arm
        assert stage.GetDefaultPrim().GetName() == arm, stage.GetDefaultPrim()
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

        # The USD is authored at qpos0, so that is the pose it must reproduce.
        data = mujoco.MjData(model)
        data.qpos[:] = 0
        mujoco.mj_kinematics(model, data)
        paths = {0: f'/{arm}'}
        for index in range(1, model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
            paths[index] = f'{paths[model.body_parentid[index]]}/{name}'

        cache = UsdGeom.XformCache()
        worst_link = 0.0
        for index in range(1, model.nbody):
            prim = stage.GetPrimAtPath(paths[index])
            assert prim, (arm, paths[index])
            world = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            worst_link = max(worst_link, float(np.abs(np.array(world) - data.xpos[index]).max()))

        # Every display geom must appear as a mesh prim at the geom's own pose.
        worst_geom, placed = 0.0, 0
        for body in range(1, model.nbody):
            geoms = [g for g in range(model.ngeom)
                     if model.geom_bodyid[g] == body
                     and model.geom_group[g] == 2
                     and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
            if not geoms:
                continue
            children = [c for c in stage.GetPrimAtPath(paths[body]).GetChildren()
                        if c.GetTypeName() == 'Mesh']
            assert len(children) == len(geoms), (arm, paths[body], len(children), len(geoms))
            rotation = data.xmat[body].reshape(3, 3)
            want = sorted(tuple(np.round(data.xpos[body] + rotation @ model.geom_pos[g], 9))
                          for g in geoms)
            got = sorted(tuple(np.round(np.array(
                cache.GetLocalToWorldTransform(c).ExtractTranslation()), 9)) for c in children)
            for expected, actual in zip(want, got):
                worst_geom = max(worst_geom, float(np.abs(np.array(expected) - np.array(actual)).max()))
                placed += 1

        meshes = [p for p in stage.Traverse() if p.GetTypeName() == 'Mesh']
        display_geoms = sum(1 for g in range(model.ngeom)
                            if model.geom_group[g] == 2 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
        assert len(meshes) == display_geoms, (arm, len(meshes), display_geoms)
        for prim in meshes:
            mesh = UsdGeom.Mesh(prim)
            assert mesh.GetPointsAttr().Get(), (arm, prim.GetPath())
            assert mesh.GetFaceVertexCountsAttr().Get(), (arm, prim.GetPath())
        assert worst_link < 1e-9 and worst_geom < 1e-9, (arm, worst_link, worst_geom)
    return {'usd_meshes': display_geoms, 'placed_geoms': placed,
            'usd_files': sorted(p.name for p in (ARMS / arm).iterdir()),
            'worst_link_error_m': worst_link, 'worst_geom_error_m': worst_geom}


def main():
    result = {}
    for arm in NAMES:
        directory = ARMS / arm
        assert (directory / 'arm.xml').is_file(), directory
        for required in ('source.json', 'UPSTREAM_LICENSE', 'README.md'):
            assert (directory / required).is_file(), (directory, required)
        model, mujoco_result = check_mujoco(arm)
        result[arm] = {**mujoco_result, **check_usd(arm, model)}
        print(arm, json.dumps(result[arm]), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print('wrote', OUT)


if __name__ == '__main__':
    sys.exit(main())
