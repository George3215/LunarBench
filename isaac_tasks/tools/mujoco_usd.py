"""离线转换编译后的 MuJoCo 模型。无 Isaac Sim 启动依赖。

显式支持 hinge/free、固定连接、三角网格/box/sphere/plane、双关节线性 tendon。
无质量的固定节点归并到上游刚体；固定世界基座的惯量不参与动力学。
新任务提供模型、初始 MjData、自由物体名称和相机名称即可复用。
"""

import hashlib

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation


def quat(q):
    return Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:])))


def pose(prim, pos, q):
    x = UsdGeom.Xformable(prim)
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp().Set(quat(q))


def axis_frame(axis):
    x = np.asarray(axis)
    seed = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(seed, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def q_from_mat(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return quat(q[[3, 0, 1, 2]])


def export_model(m, initial, output, names, camera_names):
    mujoco.mj_saveLastXML(str(output / "source.xml"), m)
    source_hash = hashlib.sha256((output / "source.xml").read_bytes()).hexdigest()
    rest = mujoco.MjData(m)
    mujoco.mj_forward(m, rest)
    stage = Usd.Stage.CreateNew(str(output / "scene.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(root.GetPrim())
    robot = UsdGeom.Xform.Define(stage, "/Scene/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    UsdGeom.Xform.Define(stage, "/Scene/Stones")
    UsdGeom.Xform.Define(stage, "/Scene/Static")
    stone_ids = {m.body(n).id for n in names}
    paths = {0: "/Scene/Static"}
    owners = {0: 0}
    for b in range(1, m.nbody):
        parent = int(m.body_parentid[b])
        rigid = bool(m.body_mass[b] > 0 or parent == 0)
        owners[b] = b if rigid else owners[parent]
        paths[b] = (
            ("/Scene/Stones/" if b in stone_ids else "/Scene/Robot/") + m.body(b).name
            if rigid
            else paths[owners[b]] + "/" + m.body(b).name
        )
        prim = UsdGeom.Xform.Define(stage, paths[b]).GetPrim()
        if rigid:
            pose(prim, rest.xpos[b], rest.xquat[b])
        else:
            owner = owners[b]
            rotation = rest.xmat[owner].reshape(3, 3)
            local_q = Rotation.from_matrix(
                rotation.T @ rest.xmat[b].reshape(3, 3)
            ).as_quat()[[3, 0, 1, 2]]
            pose(prim, rotation.T @ (rest.xpos[b] - rest.xpos[owner]), local_q)
        if rigid:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(float(m.body_mass[b]) if m.body_mass[b] > 0 else 1.0)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, m.body_ipos[b])))
            mass.CreateDiagonalInertiaAttr(
                Gf.Vec3f(
                    *map(
                        float,
                        m.body_inertia[b] if m.body_mass[b] > 0 else [0.01, 0.01, 0.01],
                    )
                )
            )
            mass.CreatePrincipalAxesAttr(quat(m.body_iquat[b]))
    # 平面在有限工作区内导出为大平面 mesh；各环境接触通过碰撞组隔离。
    geom_report = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        typ = int(m.geom_type[g])
        path = paths[b] + f"/geom_{g}"
        if typ == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(m.geom_dataid[g])
            v = m.mesh_vert[
                m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]
            ]
            f = m.mesh_face[
                m.mesh_faceadr[mesh] : m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]
            ]
            shape = UsdGeom.Mesh.Define(stage, path)
            shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
            shape.CreateFaceVertexCountsAttr([3] * len(f))
            shape.CreateFaceVertexIndicesAttr(f.ravel().tolist())
            shape.CreateSubdivisionSchemeAttr("none")
        elif typ == mujoco.mjtGeom.mjGEOM_BOX:
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(2.0)
        elif typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            shape = UsdGeom.Sphere.Define(stage, path)
            shape.CreateRadiusAttr(float(m.geom_size[g, 0]))
        elif typ == mujoco.mjtGeom.mjGEOM_PLANE:
            shape = UsdGeom.Mesh.Define(stage, path)
            s = max(float(m.geom_size[g, 0]), 3.0)
            shape.CreatePointsAttr([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)])
            shape.CreateFaceVertexCountsAttr([4])
            shape.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        else:
            raise ValueError(f"Unsupported geom type {typ}: {m.geom(g).name}")
        prim = shape.GetPrim()
        pose(prim, m.geom_pos[g], m.geom_quat[g])
        if typ == mujoco.mjtGeom.mjGEOM_BOX:
            UsdGeom.Xformable(prim).AddScaleOp().Set(
                Gf.Vec3f(*map(float, m.geom_size[g]))
            )
        # 法线贴图不作为资产依赖；使用源 RGBA 的漫反射色，保留原网格形状。
        color = m.geom_rgba[g]
        if m.geom_matid[g] >= 0:
            color = m.mat_rgba[m.geom_matid[g]]
        shape.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color[:3]))])
        shape.CreateDisplayOpacityAttr([float(color[3])])
        if color[3] == 0:
            shape.CreateVisibilityAttr("invisible")
        collision = bool(m.geom_contype[g] or m.geom_conaffinity[g])
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            if typ == mujoco.mjtGeom.mjGEOM_MESH:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                    "convexHull"
                )
            mat = UsdShade.Material.Define(stage, f"/Scene/Materials/physics_{g}")
            physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            fr = float(m.geom_friction[g, 0])
            physics.CreateStaticFrictionAttr(fr)
            physics.CreateDynamicFrictionAttr(fr)
            physics.CreateRestitutionAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
        geom_report.append(
            {
                "id": g,
                "name": m.geom(g).name,
                "body": m.body(b).name,
                "path": path,
                "collision": collision,
                "friction": m.geom_friction[g].tolist(),
            }
        )
    # 保留碰撞位掩码和显式 body 排除；同一刚体内部由 PhysX 自动排除。
    collision_geoms = [g for g in geom_report if g["collision"]]
    excluded = set(map(int, m.exclude_signature))
    for left in collision_geoms:
        a = left["id"]
        body_a = int(m.geom_bodyid[a])
        for right in collision_geoms:
            b = right["id"]
            body_b = int(m.geom_bodyid[b])
            if b <= a:
                continue
            enabled = (int(m.geom_contype[a]) & int(m.geom_conaffinity[b])) or (
                int(m.geom_contype[b]) & int(m.geom_conaffinity[a])
            )
            signature = (min(body_a, body_b) << 16) + max(body_a, body_b)
            if not enabled or signature in excluded:
                UsdPhysics.FilteredPairsAPI.Apply(
                    stage.GetPrimAtPath(left["path"])
                ).CreateFilteredPairsRel().AddTarget(right["path"])
    joints = {}
    for b in range(1, m.nbody):
        if b in stone_ids or owners[b] != b:
            continue
        parent = owners[int(m.body_parentid[b])]
        n = int(m.body_jntnum[b])
        j = int(m.body_jntadr[b])
        if n > 1:
            raise ValueError("Multiple joints in one body require explicit conversion")
        name = m.joint(j).name if n else m.body(b).name + "_fixed"
        path = "/Scene/Robot/Joints/" + name
        if n:
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Only hinge robot joints are supported")
            joint = UsdPhysics.RevoluteJoint.Define(stage, path)
            joint.CreateAxisAttr("X")
            for a in range(m.nu):
                if int(m.actuator_trnid[a, 0]) == j:
                    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
                    drive.CreateTypeAttr("force")
                    drive.CreateStiffnessAttr(
                        float(m.actuator_gainprm[a, 0] * np.pi / 180)
                    )
                    drive.CreateDampingAttr(
                        float(-m.actuator_biasprm[a, 2] * np.pi / 180)
                    )
                    drive.CreateMaxForceAttr(
                        float(max(abs(v) for v in m.actuator_forcerange[a])) or 300.0
                    )
            joint.CreateLowerLimitAttr(float(np.rad2deg(m.jnt_range[j, 0])))
            joint.CreateUpperLimitAttr(float(np.rad2deg(m.jnt_range[j, 1])))
            anchor = rest.xanchor[j]
            frame = axis_frame(rest.xaxis[j])
            joints[name] = {
                "id": j,
                "body": m.body(b).name,
                "position": float(initial.qpos[m.jnt_qposadr[j]]),
                "range": m.jnt_range[j].tolist(),
            }
        else:
            joint = UsdPhysics.FixedJoint.Define(stage, path)
            anchor = rest.xpos[b]
            frame = rest.xmat[b].reshape(3, 3)
        if parent:
            joint.CreateBody0Rel().SetTargets([paths[parent]])
        joint.CreateBody1Rel().SetTargets([paths[b]])
        for side, body in [(0, parent), (1, b)]:
            rotation = rest.xmat[body].reshape(3, 3)
            local_pos = rotation.T @ (anchor - rest.xpos[body])
            local_rot = rotation.T @ frame
            getattr(joint, f"CreateLocalPos{side}Attr")(
                Gf.Vec3f(*map(float, local_pos))
            )
            getattr(joint, f"CreateLocalRot{side}Attr")(q_from_mat(local_rot))
    mimic = []
    for k in range(m.neq):
        if m.eq_type[k] != mujoco.mjtEq.mjEQ_TENDON or m.eq_obj2id[k] != -1:
            raise ValueError("Unsupported equality")
        t = int(m.eq_obj1id[k])
        a = int(m.tendon_adr[t])
        n = int(m.tendon_num[t])
        if n != 2 or not np.all(m.wrap_type[a : a + n] == mujoco.mjtWrap.mjWRAP_JOINT):
            raise ValueError("Only two-joint fixed tendon supported")
        coeff = m.wrap_prm[a : a + 2]
        ids = m.wrap_objid[a : a + 2]
        mimic.append(
            {
                "reference": m.joint(int(ids[0])).name,
                "joint": m.joint(int(ids[1])).name,
                "gearing": float(coeff[0] / coeff[1]),
            }
        )
    cameras = {}
    for name in camera_names:
        i = m.camera(name).id
        b = int(m.cam_bodyid[i])
        path = paths[b] + "/camera_" + name
        camera = UsdGeom.Camera.Define(stage, path)
        pose(camera.GetPrim(), m.cam_pos[i], m.cam_quat[i])
        UsdGeom.Xformable(camera).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(1.0)
        )
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(aperture * 0.75)
        camera.CreateFocalLengthAttr(
            float(aperture * 0.75 / (2 * np.tan(np.deg2rad(m.cam_fovy[i]) / 2)))
        )
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        cameras[name] = {
            "body": m.body(b).name,
            "path": path,
            "position": m.cam_pos[i].tolist(),
            "quaternion_wxyz": m.cam_quat[i].tolist(),
            "fovy_deg": float(m.cam_fovy[i]),
            "initial_position": initial.cam_xpos[i].tolist(),
            "initial_quaternion_wxyz": Rotation.from_matrix(
                initial.cam_xmat[i].reshape(3, 3)
            )
            .as_quat()[[3, 0, 1, 2]]
            .tolist(),
        }
    stage.GetRootLayer().Save()
    return (
        {
            "source_sha256": source_hash,
            "joints": joints,
            "mimic": mimic,
            "cameras": cameras,
            "geoms": geom_report,
        },
        owners,
        rest,
    )
