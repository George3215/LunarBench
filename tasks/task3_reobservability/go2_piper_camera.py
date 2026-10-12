"""MuJoCo RGB-D adapter for the upstream Go2+Piper model.

The upstream model contains the robot geometry and proprioceptive sensors, but
does not define an RGB-D camera. This module injects a fixed forward camera at
load time so TASK3 can use the model's actual Go2+Piper body without modifying
the vendored checkout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

try:
    from .scenes import CAMERA_OFFSET_XYZ
    from .terrain import STAIRCASE10_RISE_M, STAIRCASE10_STEPS, STAIRCASE10_TREAD_M
except ImportError:
    from scenes import CAMERA_OFFSET_XYZ
    from terrain import STAIRCASE10_RISE_M, STAIRCASE10_STEPS, STAIRCASE10_TREAD_M


ROOT = Path(__file__).resolve().parents[2]
GO2_PIPER_XML = ROOT / "third_party" / "LeggedManip_Lab" / "mujoco" / "robots" / "go2_piper" / "go2piper.xml"
CAMERA_NAME = "task3_go2_piper_front_camera"
TARGET_NAME = "task3_target"
# Visual radius of the target marker.  It does not enter the geometric oracle,
# which casts a ray at the anchor point.
TARGET_RADIUS_M = 0.15
# The anchor is treated as the *topmost* point of the target, and the marker
# sphere is drawn hanging below it.  Drawing the sphere centred on the anchor
# instead makes the rendered evidence disagree with the oracle: on rough ground
# a 0.15 m sphere peeks over a 0.10 m ledge almost everywhere the anchor ray is
# blocked, so the label says "hidden" while the image shows a red dot.  With the
# anchor on top, an occluded anchor implies an occluded marker.


@dataclass
class Go2PiperCameraScene:
    model: object
    data: object
    renderer: object
    camera_name: str = CAMERA_NAME


def asset_status() -> dict:
    """Return an auditable status for the vendored Go2+Piper XML assets."""
    if not GO2_PIPER_XML.exists():
        return {"present": False, "reason": "go2_piper_xml_missing", "xml": str(GO2_PIPER_XML)}
    text = GO2_PIPER_XML.read_text(encoding="utf-8")
    mesh_files = []
    for line in text.splitlines():
        marker = 'file="assets/'
        if marker in line:
            mesh_files.append(line.split(marker, 1)[1].split('"', 1)[0])
    missing = [str(GO2_PIPER_XML.parent / "assets" / name) for name in mesh_files
               if not (GO2_PIPER_XML.parent / "assets" / name).exists()]
    return {
        "present": not missing,
        "xml": str(GO2_PIPER_XML),
        "mesh_count": len(mesh_files),
        "missing_mesh_count": len(missing),
        "missing_meshes": missing[:8],
        "upstream_camera_declared": "<camera" in text,
    }


@dataclass
class GroundSpec:
    """How the sampled terrain becomes a MuJoCo collision/render surface.

    ``kind`` records which representation is used, because the two are not
    equivalent in physics: MuJoCo's height-field collider was measured to stop
    the Go2+Piper from walking at all in this scene, so the staircase uses
    explicit box steps whose top faces coincide with ``Terrain.height``.
    """

    kind: str
    asset_xml: str
    geom_xml: str
    hfield_lo: float | None = None
    hfield_span: float | None = None


def stair_geoms(terrain, z_floor: float = -0.5) -> str:
    """Exact box steps for the ``staircase10`` profile.

    One box for the approach apron, one per step and one for the top landing.
    Every top face equals the height ``Terrain.height`` reports, so the oracle
    and the contact surface describe the same staircase.  The step geometry is
    read back from ``terrain.spec``, so a terrain regenerated with different
    ``step_rise`` / ``step_tread`` stays consistent with its own boxes.
    """
    spec = terrain.spec
    steps = int(spec.get("step_count", STAIRCASE10_STEPS))
    rise = float(spec.get("step_rise_m", STAIRCASE10_RISE_M))
    tread = float(spec.get("step_tread_m", STAIRCASE10_TREAD_M))
    y_half = float(terrain.y[-1] - terrain.y[0]) / 2.0
    x_min, x_max = float(terrain.x[0]), float(terrain.x[-1])
    crest = steps * tread
    top = steps * rise
    parts = []

    def box(name, cx, sx, top_z):
        sz = (top_z - z_floor) / 2.0
        parts.append(f'    <geom name="{name}" type="box" size="{sx:.6f} {y_half:.6f} {sz:.6f}" '
                     f'pos="{cx:.6f} 0 {(z_floor + sz):.6f}" rgba="0.45 0.45 0.50 1" />')

    box("task3_apron", x_min / 2.0, -x_min / 2.0, 0.0)
    for i in range(steps):
        box(f"task3_step_{i + 1}", (i + 0.5) * tread, tread / 2.0, (i + 1) * rise)
    box("task3_landing", (crest + x_max) / 2.0, (x_max - crest) / 2.0, top)
    return "\n".join(parts) + "\n"


def flat_geoms(terrain, z_floor: float = -0.5) -> str:
    """Single box slab for a level terrain; used to isolate the hfield collider."""
    x_half = float(terrain.x[-1] - terrain.x[0]) / 2.0
    y_half = float(terrain.y[-1] - terrain.y[0]) / 2.0
    top = float(terrain.heights.max())
    sz = (top - z_floor) / 2.0
    return (f'    <geom name="task3_slab" type="box" size="{x_half:.6f} {y_half:.6f} {sz:.6f}" '
            f'pos="0 0 {(z_floor + sz):.6f}" rgba="0.45 0.45 0.50 1" />\n')


BOX_GROUND_PROFILES = {"staircase10": stair_geoms, "flat": flat_geoms}


def ground_spec(terrain, kind: str = "auto") -> GroundSpec:
    """Choose the MuJoCo ground representation for a sampled terrain.

    ``slabs`` is the exact path used by the re-observability benchmark: the
    terrain is already a list of boxes, so the contact surface and the oracle
    are the same object.  ``boxes`` serves the legacy procedural profiles.
    ``hfield`` is the generic sampled-DEM path; MuJoCo's height-field collider
    was measured to stop both vendored policies from walking, so it is not
    physics-validated.
    """
    if kind == "auto" and hasattr(terrain, "slab_geoms"):
        return GroundSpec(kind="slabs", asset_xml="", geom_xml=terrain.slab_geoms())
    lo = float(terrain.heights.min())
    hi = float(terrain.heights.max())
    span = max(hi - lo, 1e-3)
    x_span = float(terrain.x[-1] - terrain.x[0])
    y_span = float(terrain.y[-1] - terrain.y[0])
    if kind == "auto":
        kind = "boxes" if terrain.name in BOX_GROUND_PROFILES else "hfield"
    if kind == "slabs":
        if not hasattr(terrain, "slab_geoms"):
            raise ValueError(f"terrain {terrain.name!r} is not a slab terrain")
        return GroundSpec(kind="slabs", asset_xml="", geom_xml=terrain.slab_geoms())
    if kind == "boxes":
        builder = BOX_GROUND_PROFILES.get(terrain.name)
        if builder is None:
            raise ValueError(f"no exact box ground for terrain profile {terrain.name!r}; "
                             f"available: {sorted(BOX_GROUND_PROFILES)}")
        return GroundSpec(kind="boxes", asset_xml="", geom_xml=builder(terrain))
    hfield = (f'    <hfield name="task3_terrain" nrow="{terrain.heights.shape[0]}" '
              f'ncol="{terrain.heights.shape[1]}" size="{x_span / 2:.6f} '
              f'{y_span / 2:.6f} {span:.6f} 0.05" />\n')
    return GroundSpec(kind="hfield", asset_xml=hfield,
                      geom_xml=f'    <geom name="task3_terrain" type="hfield" '
                               f'hfield="task3_terrain" pos="0 0 {lo:.6f}" />\n',
                      hfield_lo=lo, hfield_span=span)


def _scene_xml(terrain, target: np.ndarray, ground: str = "auto") -> tuple[str, GroundSpec]:
    """Inject terrain, target, and a body-mounted camera into Go2+Piper XML."""
    source = GO2_PIPER_XML.read_text(encoding="utf-8")
    # The upstream XML combines meshdir="assets" with file="assets/...".
    # When loaded from a string MuJoCo would resolve that as assets/assets/...;
    # keeping the file paths and clearing meshdir preserves the checkout layout.
    source = source.replace('<compiler angle="radian" meshdir="assets"',
                            '<compiler angle="radian" meshdir=""', 1)
    asset_root = (GO2_PIPER_XML.parent / "assets").as_posix()
    source = source.replace('file="assets/', f'file="{asset_root}/')
    spec = ground_spec(terrain, ground)
    # Navigation setup may add a documented finite occluder without replacing
    # the original terrain/collider. Other Task3 callers leave this absent.
    spec.geom_xml += terrain.spec.get('navigation_extra_geom_xml', '')
    source = source.replace("  <asset>\n", "  <asset>\n" + spec.asset_xml + terrain.spec.get("navigation_extra_asset_xml", ""), 1)
    source = source.replace(
        '    <body name="base_link" pos="0 -10 0.5" euler="0 0.0  1.57"',
        '    <body name="base_link" pos="0 0 0.5" euler="0 0 0"',
        1,
    )
    # Camera x-axis points right, y-axis points up; MuJoCo cameras look along -z,
    # so this frame points the camera along the robot's +x (forward) axis.
    camera = (
        f'      <camera name="{CAMERA_NAME}" pos="{CAMERA_OFFSET_XYZ[0]} 0 {CAMERA_OFFSET_XYZ[2]}" '
        'xyaxes="0 -1 0 0 0 1" fovy="87" />\n'
    )
    source = source.replace('      <site name="imu" pos="0 0 0.01" />\n',
                            '      <site name="imu" pos="0 0 0.01" />\n' + camera, 1)
    target_body = (
        f'    <body name="{TARGET_NAME}" pos="{target[0]:.6f} {target[1]:.6f} '
        f'{target[2] - TARGET_RADIUS_M:.6f}">\n'
        f'      <geom name="task3_target_geom" type="sphere" size="{TARGET_RADIUS_M}" '
        'rgba="1 0 0 1" contype="0" conaffinity="0" />\n'
        '    </body>\n'
    )
    source = source.replace("  <worldbody>\n", "  <worldbody>\n" + spec.geom_xml, 1)
    source = source.replace("</worldbody>", target_body + terrain.spec.get("navigation_dynamic_body_xml", "") + "  </worldbody>", 1)
    return source, spec


def build_model(terrain, target: np.ndarray, ground: str = "auto"):
    """Build the upstream Go2+Piper model with terrain, target and camera.

    Shared by the presentation-only RGB-D adapter and by the locomotion test so
    both run the identical physical scene.
    """
    import mujoco

    xml, spec = _scene_xml(terrain, target, ground)
    model = mujoco.MjModel.from_xml_string(xml)
    if spec.kind == "hfield":
        model.hfield_data[:] = np.clip(
            (terrain.heights.ravel() - spec.hfield_lo) / spec.hfield_span, 0.0, 1.0)
    return model


def create_scene(terrain, target: np.ndarray, width: int = 640, height: int = 480,
                 ground: str = "auto") -> Go2PiperCameraScene:
    """Build the upstream Go2+Piper model with an injected RGB-D camera."""
    import mujoco

    model = build_model(terrain, target, ground)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    return Go2PiperCameraScene(model=model, data=data, renderer=renderer)


def set_robot_pose(scene: Go2PiperCameraScene, robot_xyz: np.ndarray, yaw: float) -> None:
    """Set the free Go2 base pose without advancing the physical simulation."""
    import mujoco

    scene.data.qpos[:3] = robot_xyz
    scene.data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    scene.data.qvel[:] = 0.0
    mujoco.mj_forward(scene.model, scene.data)


def render_observation(scene: Go2PiperCameraScene, save_dir: Path | None = None,
                       stem: str = "observation") -> dict:
    """Render RGB and depth, returning only camera-observable evidence."""
    scene.renderer.update_scene(scene.data, camera=scene.camera_name)
    rgb = np.asarray(scene.renderer.render()).copy()
    scene.renderer.enable_depth_rendering()
    depth = np.asarray(scene.renderer.render()).copy()
    scene.renderer.disable_depth_rendering()

    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    target_pixels = int(np.count_nonzero((red > 120) & (red > green * 1.5) & (red > blue * 1.5)))
    result = {
        "source": "mujoco_go2_piper_mounted_camera",
        "camera_name": scene.camera_name,
        "rgb_available": True,
        "depth_available": True,
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "target_pixel_count": target_pixels,
        "target_visible_observed": target_pixels >= 3,
        "depth_finite_fraction": float(np.isfinite(depth).mean()),
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = save_dir / f"{stem}_rgb.npy"
        depth_path = save_dir / f"{stem}_depth.npy"
        np.save(rgb_path, rgb)
        np.save(depth_path, depth)
        result["rgb_path"] = str(rgb_path)
        result["depth_path"] = str(depth_path)
    return result


def close_scene(scene: Go2PiperCameraScene) -> None:
    close = getattr(scene.renderer, "close", None)
    if close is not None:
        close()
