"""Build self-contained UR / SO101 arm assets from pinned upstream descriptions.

Direct github.com is unreachable on this machine, so clones go through the
`ghfast.top` mirror (override with `LUNARBENCH_GIT_MIRROR`). Only the two
upstreams below are fetched, both pinned to an exact commit so a rerun
reproduces the same bytes.

    ros-industrial/universal_robot   -> ur_description   (ur10, ur10e, ur12e)
    TheRobotStudio/SO-ARM100         -> Simulation/SO101 (so101)

Per arm, under `assets/robots/<name>/`:

    urdf/                 upstream URDF, package:// rewritten to relative paths
    meshes/               upstream visual + collision meshes, plus OBJ exports
    arm.xml               fixed-base MuJoCo model, position servos, `tool_tip` site
    <name>.usdz           self-contained display USD for UE
    README.md             what was kept, what was authored, what is not claimed
    source.json           upstream, commit, per-file sha256, modifications, not_claimed
    UPSTREAM_LICENSE      upstream license text

Two upstream quirks drive the shape of this file:

1. MuJoCo cannot read the UR visual meshes as shipped (Collada `.dae`), so an
   OBJ is exported next to each one. The OBJ must not share a basename with the
   collision STL: MuJoCo keys meshes by basename, and a collision STL named
   `base.stl` next to a visual `base.obj` silently resolves *both* geoms to
   whichever loaded last, replacing the collision mesh with the dense visual
   one. Exports are therefore suffixed `_visual`.
2. MuJoCo's URDF compiler drops all visual geometry unless the URDF opts in with
   `<mujoco><compiler discardvisual="false"/></mujoco>`.

External tools resolve through `$LUNARBENCH_XACRO` / `$LUNARBENCH_BLENDER`, then
PATH, then the known local install. MuJoCo and usd-core come from the project
Python. Nothing else is installed and nothing ROS is needed at runtime.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Alongside piper/: one self-contained directory per arm (MJCF + meshes +
# provenance), not the older manipulators/manipulator/ + gripper/ split.
ARMS = ROOT / 'assets/robots'
CACHE = ROOT / '.local/upstream'

MIRROR = os.environ.get('LUNARBENCH_GIT_MIRROR', 'https://ghfast.top/https://github.com')

UR_SLUG = 'ros-industrial/universal_robot'
UR_COMMIT = '39ad110d8f2e8f66856a201cca88aa7a7025e3eb'
SO_SLUG = 'TheRobotStudio/SO-ARM100'
SO_COMMIT = 'eecbe3e0a9ebb23e25ad7b2759b03884c6660903'

UR_TYPES = ('ur10', 'ur10e', 'ur12e')

URDF_PACKAGE = 'ur_description'

TIMESTEP = '0.002'
GRAVITY = '0 0 -9.81'
ITERATIONS = '30'

# Servo gains for the UR arms. A URDF has no notion of a controller, so this is
# a simulation choice, and the two numbers below are the whole of it: a joint
# commands its full rated torque at this tracking error, and the derivative term
# is set for this damping ratio against the joint's own effective inertia.
#
# A single global kp cannot work here: the URDF's torque limits differ by 6x
# across the arm (330 Nm at the shoulder, 54 Nm at the wrist), so kp=4000
# saturated every wrist at 0.0135 rad of error and left them chattering against
# their stops instead of settling. Gains are therefore derived per joint.
UR_FULL_TORQUE_ERROR = 0.1
UR_DAMPING_RATIO = 1.0
UR_DAMPING = 0.5
UR_ARMATURE = 0.05
UR_HOME = [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]

VISUAL_GROUP = '2'
COLLISION_GROUP = '3'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f'{command[0]} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}')
    return result


def subprocess_env(**overrides):
    """Environment for the external tools, with empty variables dropped.

    This project invokes Python as `PYTHONPATH= python ...`, which leaves
    PYTHONPATH set-but-empty and drops whatever the login profile had put there.
    A distro-installed xacro is a console script whose entry point resolves
    through importlib.metadata, so it then dies with PackageNotFoundError unless
    its own site-packages is back on the path. Empty variables are therefore
    removed rather than passed through, and the tool's own site-packages is added
    when it lives inside an installed prefix (a ROS distro, typically).
    """
    environment = {key: value for key, value in os.environ.items() if value}
    environment.update(overrides)
    return environment


def with_prefix_pythonpath(binary, environment):
    """Put `binary`'s own site-packages on PYTHONPATH when it sits in a prefix."""
    prefix = Path(binary).resolve().parent.parent
    candidates = sorted(prefix.glob('lib/python3.*/site-packages'))
    # A ROS distro keeps its own packages under local/lib/.../dist-packages.
    candidates += sorted(prefix.glob('local/lib/python3.*/dist-packages'))
    if not candidates:
        return environment
    existing = environment.get('PYTHONPATH', '')
    parts = [str(path) for path in candidates] + ([existing] if existing else [])
    return {**environment, 'PYTHONPATH': os.pathsep.join(parts)}


def tool(name, env_var, fallbacks):
    override = os.environ.get(env_var)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(f'{name} not found; set {env_var}')


def xacro_bin():
    return tool('xacro', 'LUNARBENCH_XACRO', ['/opt/ros/humble/bin/xacro'])


def blender_bin():
    return tool('blender', 'LUNARBENCH_BLENDER', [str(Path.home() / '.local/bin/blender')])


def fetch(slug, commit):
    """Clone `slug` at `commit` into the cache; reruns reuse the checkout."""
    destination = CACHE / slug.split('/')[-1]
    if not destination.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--filter=blob:none', '--no-checkout', f'{MIRROR}/{slug}.git', str(destination)])
    marker = destination / '.pinned_commit'
    if not marker.exists() or marker.read_text().strip() != commit:
        run(['git', 'fetch', '--depth', '1', 'origin', commit], cwd=destination)
        run(['git', 'checkout', '--force', '--detach', commit], cwd=destination)
        marker.write_text(commit + '\n')
    head = run(['git', 'rev-parse', 'HEAD'], cwd=destination).stdout.strip()
    if head != commit:
        raise RuntimeError(f'{slug}: expected {commit}, checkout is {head}')
    return destination


def ament_prefix(package, package_path):
    """`$(find <package>)` resolves through a synthetic ament index.

    Nothing ROS runs at runtime. The index only maps the package name onto the
    pinned clone, so xacro expands against upstream rather than /opt/ros.
    """
    prefix = CACHE / f'ament_prefix_{package}'
    share = prefix / 'share'
    (share / 'ament_index/resource_index/packages').mkdir(parents=True, exist_ok=True)
    link = share / package
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(package_path).resolve())
    (share / 'ament_index/resource_index/packages' / package).touch()
    return prefix


_BLENDER_SCRIPT = '''\
import bpy, sys
src, dst = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.collada_import(filepath=src)
bpy.ops.wm.obj_export(filepath=dst, export_materials=False,
                      export_uv=True, export_normals=True,
                      forward_axis='Y', up_axis='Z')
'''


def export_visual_obj(dae_paths, destination):
    """Collada -> OBJ with Blender, preserving the Z-up frame MuJoCo expects.

    `obj_export` defaults to a Y-up axis conversion, which silently rotates every
    link by 90 degrees. `forward_axis='Y', up_axis='Z'` keeps the source frame;
    the result then agrees with the upstream collision STL bounds.

    Names are suffixed `_visual` so they cannot collide with a collision STL of
    the same basename inside MuJoCo's mesh table.
    """
    destination.mkdir(parents=True, exist_ok=True)
    script = destination / '_dae_to_obj.py'
    script.write_text(_BLENDER_SCRIPT)
    produced = {}
    for dae in sorted(dae_paths):
        obj = destination / f'{Path(dae).stem}_visual.obj'
        if not obj.exists():
            run([blender_bin(), '--background', '--python', str(script), '--', str(dae), str(obj)])
        produced[Path(dae).stem] = obj
    return produced


def parse_urdf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        def attr(name, default):
            return [float(v) for v in (origin.get(name, default).split() if origin is not None else default.split())]
        limit = joint.find('limit')
        joints.append({
            'name': joint.get('name'),
            'type': joint.get('type'),
            'parent': joint.find('parent').get('link'),
            'child': joint.find('child').get('link'),
            'xyz': attr('xyz', '0 0 0'),
            'rpy': attr('rpy', '0 0 0'),
            'lower': float(limit.get('lower', 0.0)) if limit is not None else 0.0,
            'upper': float(limit.get('upper', 0.0)) if limit is not None else 0.0,
            'effort': float(limit.get('effort', 0.0)) if limit is not None else 0.0,
        })
    return joints


def fixed_chain_pose(joints, start, target):
    """Compose the fixed-joint chain `start` -> `target` into (pos, quat_wxyz)."""
    import mujoco
    position = np.zeros(3)
    rotation = np.eye(3)
    current = start
    while current != target:
        step = next((j for j in joints if j['type'] == 'fixed' and j['parent'] == current), None)
        if step is None:
            raise RuntimeError(f'no fixed joint from {current!r} towards {target!r}')
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array(step['rpy']), 'xyz')
        flat = np.zeros(9)
        mujoco.mju_quat2Mat(flat, quat)
        local = flat.reshape(3, 3)
        position = np.asarray(step['xyz']) + local @ position
        rotation = local @ rotation
        current = step['child']
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.reshape(9))
    return position, quat


def geom_groups(root):
    """Sort compiled geoms into display (group 2) and collision (group 3).

    MuJoCo's URDF importer already marks visual geoms `contype="0"
    conaffinity="0"`; that flag is what distinguishes them, not the mesh name.
    """
    containers = [root.find('worldbody')] + list(root.iter('body'))
    for geom in (g for c in containers if c is not None for g in c.findall('geom')):
        visual = geom.get('contype') == '0' and geom.get('conaffinity') == '0' and geom.get('mesh')
        geom.set('group', VISUAL_GROUP if visual else COLLISION_GROUP)


def compile_mjcf(urdf_for_mujoco, destination, meshdir, model_name):
    """Compile the URDF with MuJoCo and restate it as a standalone MJCF.

    MuJoCo's URDF compiler already produces correct kinematics, inertials, joint
    ranges and `actuatorfrcrange`. It cannot express actuators or sites, so those
    are added onto the compiled tree instead of hand-authoring the whole model.
    """
    import mujoco
    scratch = Path(tempfile.mkdtemp(prefix='mjcf-'))
    compiled = scratch / 'compiled.xml'
    mujoco.mj_saveLastXML(str(compiled), mujoco.MjModel.from_xml_path(str(urdf_for_mujoco)))
    tree = ET.parse(compiled)
    root = tree.getroot()
    root.set('model', model_name)
    root.remove(root.find('compiler'))

    mesh_root = (destination.parent / meshdir).resolve()
    for mesh in root.find('asset').findall('mesh'):
        mesh.set('file', str(Path(mesh.get('file')).resolve().relative_to(mesh_root)))

    ET.ElementTree(root).write(destination, encoding='unicode')
    shutil.rmtree(scratch, ignore_errors=True)
    return destination


def servo_gains(model, full_torque_error, damping_ratio):
    """Per-actuator (kp, kv) from the joint's torque limit and effective inertia.

    kp is chosen so a joint reaches its rated torque at `full_torque_error`,
    which keeps the servo from saturating on ordinary tracking error. kv then
    follows from critical damping against the diagonal of the mass matrix at
    qpos0, the configuration the model is authored around.
    """
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    gains = []
    for actuator in range(model.nu):
        dof = model.jnt_dofadr[model.actuator_trnid[actuator, 0]]
        effort = float(np.abs(model.actuator_forcerange[actuator]).min())
        kp = effort / full_torque_error
        gains.append((kp, 2 * damping_ratio * float(np.sqrt(kp * mass[dof, dof]))))
    return gains


def promote_root_link(root):
    """Re-home the compiled root link's geometry into a real `base_link` body.

    MuJoCo's URDF compiler welds a fixed root link into `worldbody`, which drops
    the link name and, worse, exempts the pair from `filterparent`. The UR base
    and shoulder collision meshes overlap by 0.3 mm, so as separate bodies the
    overlap is filtered as an ordinary parent/child pair, but with the base
    welded to the world the shoulder pan joint carried four standing contacts at
    the home pose. Offsets are untouched: the body sits at the world origin.
    """
    world = root.find('worldbody')
    welded = [child for child in world if child.tag in ('geom', 'inertial')]
    if not welded:
        return
    base = ET.Element('body', {'name': 'base_link'})
    for child in welded:
        world.remove(child)
        base.append(child)
    world.insert(0, base)


def finalize_ur_mjcf(mjcf, joints, tool_body, tool_pose, meshdir, home):
    tree = ET.parse(mjcf)
    root = tree.getroot()
    promote_root_link(root)
    tree = _with_scaffold(root, meshdir, f'{mjcf.parent.name} fixed base task adapter')

    geom_groups(root)

    bodies = {body.get('name'): body for body in root.iter('body')}
    position, quat = tool_pose
    ET.SubElement(bodies[tool_body], 'site', {
        'name': 'tool_tip',
        'pos': _floats(position),
        'quat': _floats(quat),
        'size': '.008',
        'rgba': '0 1 0 1',
    })

    actuator = ET.SubElement(root, 'actuator')
    for joint in joints:
        ET.SubElement(actuator, 'position', {
            'name': joint['name'],
            'joint': joint['name'],
            'ctrlrange': f"{joint['lower']:.10g} {joint['upper']:.10g}",
            'forcerange': f"{-joint['effort']:.10g} {joint['effort']:.10g}",
        })
    # The gains depend on the compiled inertias, so they are filled in on a
    # second pass once the model can be loaded.
    _write(tree, mjcf)
    import mujoco
    gains = servo_gains(mujoco.MjModel.from_xml_path(str(mjcf)), UR_FULL_TORQUE_ERROR, UR_DAMPING_RATIO)
    for element, (kp, kv) in zip(root.find('actuator').findall('position'), gains):
        element.set('kp', f'{kp:.10g}')
        element.set('kv', f'{kv:.10g}')
    # Upstream's base and shoulder collision meshes overlap by 0.3 mm, which
    # left four standing contacts on the shoulder pan joint: the home keyframe
    # drifted 0.04 rad in one second with zero control error. The overlap is in
    # the UR's own simplified collision geometry, at a housing interface that is
    # internal to the real arm, so the pair is excluded rather than remeshed.
    contact = ET.Element('contact')
    ET.SubElement(contact, 'exclude', {'body1': 'base_link', 'body2': 'shoulder_link'})
    root.insert(list(root).index(root.find('actuator')), contact)

    key = ET.SubElement(ET.SubElement(root, 'keyframe'), 'key', {'name': 'home'})
    key.set('qpos', _floats(home))
    key.set('ctrl', _floats(home))
    _write(tree, mjcf)


def _with_scaffold(root, meshdir, model_name):
    """Shared option/default blocks, inserted in the order MuJoCo expects."""
    root.set('model', model_name)
    compiler = ET.Element('compiler', {'angle': 'radian', 'autolimits': 'true', 'meshdir': meshdir})
    option = ET.Element('option', {'timestep': TIMESTEP, 'gravity': GRAVITY, 'iterations': ITERATIONS})
    default = ET.Element('default')
    ET.SubElement(default, 'joint', {'damping': str(UR_DAMPING), 'armature': str(UR_ARMATURE)})
    ET.SubElement(default, 'geom', {'condim': '3', 'friction': '.8 .02 .001'})
    root.insert(0, default)
    root.insert(0, option)
    root.insert(0, compiler)
    return ET.ElementTree(root)


def _floats(values):
    return ' '.join(f'{v:.10g}' for v in values)


def _write(tree, path):
    ET.indent(tree)
    tree.write(path, encoding='unicode', xml_declaration=True)


def link_hierarchy(mjcf):
    """Local transforms of every body and its display geoms, at qpos0.

    Local rather than world transforms: the USD mirrors the MJCF tree, so each
    prim carries its own offset from its parent and the base link keeps whatever
    rotation the URDF gave it.
    """
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    # MuJoCo names a mesh after its `name` if present and after the file stem
    # otherwise; the upstream SO-101 MJCF omits `name` entirely.
    names = {}
    for mesh in ET.parse(mjcf).getroot().find('asset').findall('mesh'):
        names[mesh.get('name') or Path(mesh.get('file')).stem] = mesh

    def body_name(index):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)

    links = []
    for index in range(1, model.nbody):
        parent = model.body_parentid[index]
        links.append({
            'name': body_name(index),
            'parent': body_name(parent) if parent else None,
            'transform': transform(model.body_pos[index], model.body_quat[index]),
            'meshes': [],
        })
    by_body = {link['name']: link for link in links}
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[geom] != int(VISUAL_GROUP):
            continue
        # Display geometry welded into the world body would have no link to hang
        # from; `promote_root_link` is what keeps that from happening.
        assert model.geom_bodyid[geom], 'display geom in worldbody'
        owner = body_name(model.geom_bodyid[geom])
        source = names.get(model.mesh(model.geom_dataid[geom]).name)
        by_body[owner]['meshes'].append({
            'file': Path(source.get('file')).name if source is not None else None,
            'subdir': str(Path(source.get('file')).parent) if source is not None else '',
            'transform': transform(model.geom_pos[geom], model.geom_quat[geom]),
            # `geom_rgba` stays at MuJoCo's default when the geom names a
            # material, so the material's colour is what the viewer shows.
            'color': [float(c) for c in (
                model.mat_rgba[model.geom_matid[geom]] if model.geom_matid[geom] >= 0
                else model.geom_rgba[geom])[:3]],
        })
    return links


def transform(position, quat):
    """Row-major 4x4 from a MuJoCo position and wxyz quaternion."""
    import mujoco
    flat = np.zeros(9)
    mujoco.mju_quat2Mat(flat, np.asarray(quat, dtype=float))
    matrix = np.eye(4)
    matrix[:3, :3] = flat.reshape(3, 3)
    matrix[:3, 3] = position
    return matrix.tolist()


def read_stl(path):
    """Read a binary or ASCII STL into (points, triangles)."""
    raw = Path(path).read_bytes()
    if raw[:5].lower() == b'solid' and b'facet' in raw[:2048].lower():
        points, faces = [], []
        for line in raw.decode('ascii', 'ignore').splitlines():
            parts = line.split()
            if parts and parts[0] == 'vertex':
                points.append([float(v) for v in parts[1:4]])
        for index in range(0, len(points) - 2, 3):
            faces.append([index, index + 1, index + 2])
        return points, faces
    import struct
    count = struct.unpack('<I', raw[80:84])[0]
    records = np.frombuffer(raw[84:84 + count * 50], dtype=np.uint8).reshape(count, 50)
    triangles = records[:, 12:48].copy().view('<f4').reshape(count, 3, 3).astype(float)
    return triangles.reshape(-1, 3).tolist(), [[i, i + 1, i + 2] for i in range(0, count * 3, 3)]


def read_obj(path):
    """Read an OBJ into (points, triangles, normals, uvs, face_uv)."""
    points, normals, uvs, faces, face_uv = [], [], [], [], []
    for line in Path(path).read_text(errors='ignore').splitlines():
        if line.startswith('v '):
            points.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('vn '):
            normals.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('vt '):
            uvs.append([float(v) for v in line.split()[1:3]])
        elif line.startswith('f '):
            corners = [c.split('/') for c in line.split()[1:]]
            indices = [int(c[0]) - 1 for c in corners]
            for offset in range(1, len(indices) - 1):
                faces.append([indices[0], indices[offset], indices[offset + 1]])
                for corner in (corners[0], corners[offset], corners[offset + 1]):
                    face_uv.append(int(corner[1]) - 1 if len(corner) > 1 and corner[1] else 0)
    return points, faces, normals, uvs, face_uv


def mesh_to_usd_mesh(stage, path, name, color):
    """Author a UsdGeomMesh from an OBJ or STL source.

    USD cannot reference OBJ or STL, and a `.usdz` is meant to be self-contained,
    so the geometry is inlined rather than linked.
    """
    from pxr import Gf, UsdGeom, Vt
    if Path(path).suffix.lower() == '.stl':
        points, faces = read_stl(path)
        normals, uvs, face_uv = [], [], []
    else:
        points, faces, normals, uvs, face_uv = read_obj(path)
    mesh = UsdGeom.Mesh.Define(stage, name)
    # The geometry is already a dense triangulation, and catmullClark is USD's
    # default; declaring `none` keeps a viewer or importer from subdividing it.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in faces for i in f]))
    if normals:
        mesh.CreateNormalsAttr(Vt.Vec3fArray(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uvs:
        mesh.CreatePrimvar('st', UsdGeom.Tokens.faceVarying, Vt.Vec2fArray(uvs))
    return mesh


def usd_matrix(transform):
    """Transpose a column-vector homogeneous matrix into Gf's row-vector one.

    `transform()` returns the usual `v' = R v + t` matrix. GfMatrix4d multiplies
    row vectors on the left (`v' = v * M`), so its rotation block is the
    transpose and its translation sits in the last row rather than the last
    column. Passing the standard form straight through silently drops every
    translation and transposes every rotation.
    """
    from pxr import Gf
    return Gf.Matrix4d(*np.asarray(transform, dtype=float).T.reshape(-1).tolist())


def build_usd(arm, links, mesh_root, destination):
    """Self-contained display USD: link hierarchy plus one mesh per link.

    UE reads this through its USD Stage importer. It carries geometry and the
    zero-configuration link frames, not a simulation and not PhysX articulation.
    Transforms are local to each link, so the prim tree mirrors the MJCF tree.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils
    scratch = Path(tempfile.mkdtemp(prefix='usd-'))
    stage = Usd.Stage.CreateNew(str(scratch / f'{arm}.usd'))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f'/{arm}')
    stage.SetDefaultPrim(root.GetPrim())
    # Bodies come out of the model in topological order, so a link's parent prim
    # always exists by the time the link is defined. `world` is the MJCF root and
    # is folded into `base_link`, the URDF link the compiled base geom belongs to.
    paths = {}
    for link in links:
        parent = paths.get(link['parent'], root.GetPath())
        path = f'{parent}/{link["name"]}'
        paths[link['name']] = path
        prim = UsdGeom.Xform.Define(stage, path)
        prim.AddTransformOp().Set(usd_matrix(link['transform']))
        for index, mesh in enumerate(link['meshes']):
            if not mesh['file']:
                continue
            source = mesh_root / mesh['subdir'] / mesh['file']
            child = mesh_to_usd_mesh(stage, source, f'{path}/mesh_{index}', mesh['color'])
            child.AddTransformOp().Set(usd_matrix(mesh['transform']))
    stage.GetRootLayer().Save()
    if destination.exists():
        destination.unlink()
    UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(scratch / f'{arm}.usd')), str(destination))
    shutil.rmtree(scratch, ignore_errors=True)
    return destination


def write_record(arm_dir, record, license_source):
    (arm_dir / 'source.json').write_text(json.dumps(record, indent=2) + '\n')
    shutil.copy2(license_source, arm_dir / 'UPSTREAM_LICENSE')


def build_ur(ur_type):
    package = fetch(UR_SLUG, UR_COMMIT) / URDF_PACKAGE
    arm_dir = ARMS / ur_type
    (arm_dir / 'urdf').mkdir(parents=True, exist_ok=True)

    urdf = arm_dir / 'urdf' / f'{ur_type}.urdf'
    ament = ament_prefix(URDF_PACKAGE, package)
    xacro = xacro_bin()
    environment = subprocess_env(AMENT_PREFIX_PATH=str(ament))
    environment.pop('ROS_PACKAGE_PATH', None)
    environment = with_prefix_pythonpath(xacro, environment)
    urdf.write_text(run([xacro, f'urdf/{ur_type}.xacro'], cwd=package, env=environment).stdout)

    # package://ur_description/meshes/<dir>/... already carries the meshes/
    # segment; the URDF sits one level down in urdf/, so only the leading ../
    # is added. <dir> is normalized to this arm's own name: ur12e reuses the
    # ur10e meshes upstream, and making every arm directory self-contained
    # keeps meshdir and the USD mesh root uniform.
    tree = ET.parse(urdf)
    sources = set()
    marker = f'package://{URDF_PACKAGE}/meshes/'
    for mesh in tree.getroot().iter('mesh'):
        if not mesh.get('filename', '').startswith(marker):
            continue
        rest = mesh.get('filename')[len(marker):]
        sources.add(rest.split('/', 1)[0])
        mesh.set('filename', f'../meshes/{ur_type}/{rest.split("/", 1)[1]}')
    assert sources, ur_type
    _write(tree, urdf)
    joints = parse_urdf(urdf)

    mesh_target = arm_dir / 'meshes' / ur_type
    if mesh_target.exists():
        shutil.rmtree(mesh_target)
    mesh_target.mkdir(parents=True)
    seen = {}
    for source in sorted(sources):
        for kind in ('visual', 'collision'):
            origin = package / 'meshes' / source / kind
            if not origin.is_dir():
                continue
            destination = mesh_target / kind
            destination.mkdir(parents=True, exist_ok=True)
            for entry in sorted(origin.iterdir()):
                # MuJoCo keys meshes by basename, so two sources contributing
                # the same filename would silently collapse into one mesh.
                if entry.name in seen and seen[entry.name] != entry:
                    raise RuntimeError(f'{ur_type}: mesh basename collision {entry.name}')
                seen[entry.name] = entry
                shutil.copy2(entry, destination / entry.name)
    export_visual_obj(sorted((mesh_target / 'visual').glob('*.dae')), mesh_target / 'visual')
    (mesh_target / 'visual' / '_dae_to_obj.py').unlink()

    # A parallel URDF points MuJoCo at the OBJ visuals and opts visual geoms in.
    # MuJoCo's URDF importer resolves relative mesh paths against the process
    # CWD rather than the URDF, so these are made absolute here and turned back
    # into meshdir-relative paths on the compiled MJCF.
    for_mujoco = arm_dir / 'urdf' / f'{ur_type}_mujoco.urdf'
    tree = ET.parse(urdf)
    for mesh in tree.getroot().iter('mesh'):
        filename = mesh.get('filename')
        if filename.endswith('.dae'):
            filename = f'{filename[:-4]}_visual.obj'
        mesh.set('filename', str((arm_dir / 'urdf' / filename).resolve()))
    optin = ET.Element('mujoco')
    ET.SubElement(optin, 'compiler', {'discardvisual': 'false'})
    tree.getroot().insert(0, optin)
    _write(tree, for_mujoco)

    mjcf = compile_mjcf(for_mujoco, arm_dir / 'arm.xml', f'meshes/{ur_type}', f'{ur_type}_robot')
    for_mujoco.unlink()
    revolute = [j for j in joints if j['type'] == 'revolute']
    finalize_ur_mjcf(mjcf, revolute, 'wrist_3_link', fixed_chain_pose(joints, 'wrist_3_link', 'tool0'),
                     f'meshes/{ur_type}', UR_HOME)

    build_usd(ur_type, link_hierarchy(mjcf), mesh_target, arm_dir / f'{ur_type}.usdz')

    write_record(arm_dir, {
        'source': f'https://github.com/{UR_SLUG}',
        'commit': UR_COMMIT,
        'package': URDF_PACKAGE,
        'xacro': f'{URDF_PACKAGE}/urdf/{ur_type}.xacro',
        'urdf': f'urdf/{ur_type}.urdf',
        'urdf_sha256': sha256(urdf),
        'arm_xml_sha256': sha256(mjcf),
        'usdz_sha256': sha256(arm_dir / f'{ur_type}.usdz'),
        'meshes': {str(p.relative_to(arm_dir)): sha256(p)
                   for p in sorted(mesh_target.rglob('*')) if p.is_file()},
        'joints': [{k: j[k] for k in ('name', 'lower', 'upper', 'effort')} for j in revolute],
        'modifications': [
            'Expanded the upstream xacro to a standalone URDF',
            'Rewrote package:// mesh URLs to paths relative to the asset directory',
            'Exported each Collada visual mesh to OBJ (Blender, Z-up preserved) because MuJoCo cannot read DAE',
            'Compiled the URDF with MuJoCo and added position servos, a tool_tip site and a home keyframe',
            'Re-homed the compiled root link into a named base_link body and excluded the '
            'base/shoulder collision pair, whose upstream meshes overlap by 0.3 mm',
            'Authored a display-only USD for UE',
        ] + ([f'Copied the meshes from upstream meshes/{sorted(sources)[0]}, whose URDF and '
              f'arm.xml this model reproduces exactly: {ur_type} ships no mesh directory of '
              f'its own and upstream gives it the same description files as '
              f'{sorted(sources)[0]}'] if sources != {ur_type} else []),
        'not_claimed': ['hardware calibration', 'trained control', 'UE-side physics simulation'],
    }, package / 'LICENSE')
    return arm_dir


def build_so101():
    source = fetch(SO_SLUG, SO_COMMIT) / 'Simulation/SO101'
    arm_dir = ARMS / 'so101'
    (arm_dir / 'urdf').mkdir(parents=True, exist_ok=True)

    urdf = arm_dir / 'urdf' / 'so101_new_calib.urdf'
    tree = ET.parse(source / 'so101_new_calib.urdf')
    for mesh in tree.getroot().iter('mesh'):
        if mesh.get('filename', '').startswith('assets/'):
            mesh.set('filename', f'../meshes/{mesh.get("filename")[len("assets/"):]}')
    _write(tree, urdf)

    mesh_target = arm_dir / 'meshes'
    if mesh_target.exists():
        shutil.rmtree(mesh_target)
    mesh_target.mkdir(parents=True)
    for stl in sorted((source / 'assets').glob('*.stl')):
        shutil.copy2(stl, mesh_target / stl.name)

    # The upstream MJCF already carries tuned servo gains and visual/collision
    # classes; it is patched in place rather than recompiled from the URDF, which
    # would drop all of that.
    mjcf = arm_dir / 'arm.xml'
    tree = ET.parse(source / 'so101_new_calib.xml')
    root = tree.getroot()
    root.find('compiler').set('meshdir', 'meshes')
    scaffold = ET.Element('option', {'timestep': TIMESTEP, 'gravity': GRAVITY, 'iterations': ITERATIONS})
    root.insert(list(root).index(root.find('compiler')) + 1, scaffold)

    joints = parse_urdf(urdf)
    tool_pose = fixed_chain_pose(joints, 'gripper_link', 'gripper_frame_link')
    bodies = {body.get('name'): body for body in root.iter('body')}
    ET.SubElement(bodies['gripper'], 'site', {
        'name': 'tool_tip',
        'pos': _floats(tool_pose[0]),
        'quat': _floats(tool_pose[1]),
        'size': '.008',
        'rgba': '0 1 0 1',
    })

    # The upstream MJCF ships no keyframe; all-default is the calibrated zero
    # pose the URDF describes, so it is named `home` to match the UR arms. The
    # sizes come from the compiled model rather than from counting <joint>
    # elements, which also counts the equality-constrained backlash joints.
    _write(tree, mjcf)
    import mujoco
    compiled = mujoco.MjModel.from_xml_path(str(mjcf))
    key = ET.SubElement(ET.SubElement(root, 'keyframe'), 'key', {'name': 'home'})
    key.set('qpos', ' '.join(['0'] * compiled.nq))
    key.set('ctrl', ' '.join(['0'] * compiled.nu))
    _write(tree, mjcf)

    build_usd('so101', link_hierarchy(mjcf), mesh_target, arm_dir / 'so101.usdz')

    revolute = [j for j in joints if j['type'] == 'revolute']
    write_record(arm_dir, {
        'source': f'https://github.com/{SO_SLUG}',
        'commit': SO_COMMIT,
        'package': 'Simulation/SO101',
        'urdf': 'urdf/so101_new_calib.urdf',
        'urdf_sha256': sha256(urdf),
        'arm_xml_sha256': sha256(mjcf),
        'usdz_sha256': sha256(arm_dir / 'so101.usdz'),
        'meshes': {str(p.relative_to(arm_dir)): sha256(p)
                   for p in sorted(mesh_target.rglob('*')) if p.is_file()},
        'joints': [{k: j[k] for k in ('name', 'lower', 'upper', 'effort')} for j in revolute],
        'modifications': [
            'Copied the upstream URDF and mesh set, rewriting mesh paths to be relative',
            'Kept the upstream MJCF as the MuJoCo model, including its servo gains and geom classes',
            'Retargeted meshdir from assets/ to meshes/ and pinned the project timestep and gravity',
            'Added a tool_tip site on the gripper at the upstream gripper_frame_link pose',
            'Authored a display-only USD for UE',
        ],
        'not_claimed': ['hardware calibration', 'trained control', 'UE-side physics simulation'],
    }, source.parent.parent / 'LICENSE')
    return arm_dir


def main():
    wanted = sys.argv[sys.argv.index('--only') + 1].split(',') if '--only' in sys.argv else None
    for ur_type in UR_TYPES:
        if wanted and ur_type not in wanted:
            continue
        print(f'building {ur_type}', flush=True)
        build_ur(ur_type)
    if not wanted or 'so101' in wanted:
        print('building so101', flush=True)
        build_so101()


if __name__ == '__main__':
    main()
