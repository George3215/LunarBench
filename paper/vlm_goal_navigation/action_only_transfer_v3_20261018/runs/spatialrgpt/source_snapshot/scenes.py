"""Scene anchors for TASK3 terrain-induced target re-observability.

A scene fixes everything that must stay identical between the candidates of one
counterfactual question: where the target stands, where the robot started when
the target was still visible, and where the robot is when the target is lost.
Only the candidate action changes afterwards.

Two constants tie the geometric oracle to the real robot and must stay in sync
with the MuJoCo model:

``ROBOT_BASE_HEIGHT_M``
    Height of the ``base_link`` origin above the local contact surface while the
    robot stands.  Measured from the headless policy rollout, not assumed.
``CAMERA_OFFSET_XYZ``
    Pose of the injected camera in ``base_link``, copied from the camera that
    :mod:`go2_piper_camera` injects into the upstream Go2+Piper XML.  The oracle
    and the rendered RGB-D must share one eye point, otherwise the privileged
    label and the camera evidence describe different cameras.

The upstream Go2+Piper has no pan/tilt joint, so ``look_up`` is not a physical
action of this robot; it exists only for the primitive ``proxy`` backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

ROBOT_BASE_HEIGHT_M = 0.30
CAMERA_OFFSET_XYZ = (0.31, 0.0, 0.16)


@dataclass(frozen=True)
class Scene:
    name: str
    terrain_profile: str
    target_xy: tuple[float, float]
    target_height_m: float
    approach_xy: tuple[float, float]
    lost_xy: tuple[float, float]
    yaw_rad: float = 0.0
    forward_m: float = 0.45
    climb_m: float = 2.40
    back_m: float = 2.40
    turn_rad: float = 0.6109  # 35 deg
    # Absent by default: the target then sits this far above the local terrain.
    target_z_absolute: float | None = None
    # Where the robot stood when it last saw the target, for paired-history scenes.
    history_xy: tuple[float, float] | None = None
    history_saw_target: bool = False
    # Keyword arguments for a generated slab terrain, when the profile name alone
    # is not enough (paired-history worlds are produced by a parameter search).
    terrain_params: dict | None = None
    # Mechanics this scene is meant to exercise, and what the honest answer is.
    mechanism: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------
# Legacy procedural profiles (sampled height field + box ground)
# --------------------------------------------------------------------------

STAIRCASE10 = Scene(
    name="staircase10",
    terrain_profile="staircase10",
    target_xy=(5.00, 0.00),
    target_height_m=0.50,
    approach_xy=(-4.60, 0.00),
    lost_xy=(-0.50, 0.00),
    mechanism="occlusion by a rising crest; climbing the crest restores line-of-sight",
    notes=("target visible from the far apron, lost near the foot of the stairs",
           "only a physical climb raises the eye above the crest again"),
)

GENERIC = Scene(
    name="generic",
    terrain_profile="generic",
    target_xy=(4.65, 0.00),
    target_height_m=0.40,
    approach_xy=(-3.00, 0.00),
    lost_xy=(0.90, 0.00),
)

# --------------------------------------------------------------------------
# Experiment 2 benchmark: slab terrains, one distinct mechanism per scene
# --------------------------------------------------------------------------

STAIRS_UP = Scene(
    name="stairs_up",
    terrain_profile="stairs_up",
    target_xy=(5.00, 0.00),
    target_height_m=0.50,
    approach_xy=(-4.60, 0.00),
    lost_xy=(-0.50, 0.00),
    mechanism="rising crest hides a target that was visible from far away",
    notes=("expect V: 1 -> 0 -> 1 while walking up",
           "expected best action: forward (commit to the climb)"),
)

STAIRS_DOWN = Scene(
    name="stairs_down",
    terrain_profile="stairs_down",
    target_xy=(4.00, 0.00),
    target_height_m=0.50,
    approach_xy=(-4.50, 0.00),
    lost_xy=(-0.50, 0.00),
    mechanism="descending staircase; walking down lowers the eye away from the occluder",
    notes=("used to show that descending terrain alone never occludes a target "
           "beyond it - see the TASK3 README"),
)

WINDOW_WALL = Scene(
    name="window_wall",
    terrain_profile="window_wall",
    target_xy=(5.00, 3.00),
    target_z_absolute=2.00,
    target_height_m=2.00,
    approach_xy=(-2.00, -2.70),
    lost_xy=(0.00, 0.00),
    mechanism="a wall pierced by a pinhole window; the reveal side flips with the target",
    notes=("pair family used by Experiment 1 (history necessity)",),
)

BARRIER = Scene(
    name="barrier",
    terrain_profile="barrier",
    target_xy=(5.00, 0.00),
    target_z_absolute=2.00,
    target_height_m=2.00,
    approach_xy=(-5.50, 0.00),
    lost_xy=(0.50, 0.00),
    mechanism="a wall segment that is finite in y; the sight line can be taken around it",
    notes=("expect V: 1 -> 0 and stay 0 for forward and back, 1 again after a lateral move",
           "expected best action: left/right, because retreat cannot outrun the wall"),
)

TRENCH = Scene(
    name="trench",
    terrain_profile="trench",
    target_xy=(4.60, 0.00),
    target_height_m=0.15,
    approach_xy=(-5.00, 0.00),
    lost_xy=(-0.30, 0.00),
    mechanism="the robot walks down into a depression; the raised far rim hides the target",
    notes=("expect V: 1 -> 0 -> 1, visible again once the robot climbs the far ramp",
           "expected best action: forward (commit through the depression)"),
)

PIT = Scene(
    name="pit",
    terrain_profile="pit",
    target_xy=(4.00, 0.00),
    target_height_m=0.30,
    approach_xy=(-5.20, 0.00),
    lost_xy=(-1.40, 0.00),
    mechanism="descending into a pit with an unclimbable far wall",
    notes=("expect V: 1 -> 0 and no forward recovery; only retreat restores sight",
           "expected best action: back"),
)

WAVES = Scene(
    name="waves",
    terrain_profile="waves",
    target_xy=(5.20, 0.00),
    target_z_absolute=0.25,
    target_height_m=0.25,
    approach_xy=(-4.00, 0.00),
    lost_xy=(-1.00, 0.00),
    mechanism="repeated gentle undulation; visibility flips many times along one path",
    notes=("expect several 1 -> 0 -> 1 transitions",
           "the best action is expected to depend on which trough the robot is in"),
)

BASIN = Scene(
    name="basin",
    terrain_profile="basin",
    # The target sits *on the floor of the depression*, not beyond a far rim.
    target_xy=(2.00, 0.00),
    target_z_absolute=-0.45,
    target_height_m=0.45,
    approach_xy=(-4.00, 0.00),
    lost_xy=(-5.80, 0.00),
    mechanism="a target inside a stepped basin; the near rim occludes it at a grazing angle",
    notes=("measured V(x) is X V X: hidden for x < -5.0, visible -4.95..+1.95, hidden after",
           "the occlusion is a grazing angle, not a wall: from very far away the sight "
           "line to a target below the rim hits the flat ground first",
           "note this is the *opposite* ordering from stairs_up: here walking toward the "
           "basin is what reveals the target, which is why the direction of the effect "
           "has to be measured rather than assumed"),
)

PERLIN = Scene(
    name="perlin",
    terrain_profile="perlin",
    target_xy=(5.00, 0.00),
    target_height_m=0.45,
    approach_xy=(2.00, 0.00),
    lost_xy=(-1.80, 0.00),
    mechanism="Perlin/fBm undulation, quantised into walkable steps",
    notes=("measured V(x) is X V X V X V X: six transitions along one traverse",
           "anchors are chosen where the oracle label and the rendered camera agree, so "
           "'hidden' means zero target pixels rather than a sliver",
           "the best action depends on whether the robot is in a trough or on a crest"),
)

PERLIN_ROUGH = Scene(
    name="perlin_rough",
    terrain_profile="perlin_rough",
    target_xy=(5.00, 0.00),
    target_height_m=0.18,
    approach_xy=(2.00, 0.00),
    lost_xy=(-0.85, 0.00),
    mechanism="rougher Perlin/fBm undulation: tighter features and a larger amplitude",
    notes=("measured V(x) is X V X V X: four transitions",
           "the clean hidden band is only 0.1 m wide here: on fine relief a short target "
           "is fully occluded only in the deepest trough, so the oracle label and the "
           "rendered camera disagree almost everywhere else",
           "stresses the walkability limit of the vendored policy"),
)

SCENES = {
    "staircase10": STAIRCASE10,
    "stairs_up": STAIRS_UP,
    "stairs_down": STAIRS_DOWN,
    "barrier": BARRIER,
    "trench": TRENCH,
    "pit": PIT,
    "waves": WAVES,
    "window_wall": WINDOW_WALL,
    "basin": BASIN,
    "perlin": PERLIN,
    "perlin_rough": PERLIN_ROUGH,
}

# Scenes that form the Experiment 2 benchmark, in reporting order.
BENCHMARK_SCENES = ["stairs_up", "trench", "pit", "barrier", "waves",
                    "basin", "perlin", "perlin_rough"]

# ---------------------------------------------------------------------------
# Terrain-induced occlusion mechanisms
# ---------------------------------------------------------------------------
#
# The benchmark is organised by *why* the target stops being visible, not by what
# the ground is called.  Each entry is one realistic situation in which a robot
# that has already seen a target loses sight of it, and the point of the taxonomy
# is that a reviewer should read it as "these are situations that happen" rather
# than "these are obstacles the authors built to break the baselines".
#
# Two framing rules follow from this and are enforced by the wording here:
#
# * these are **terrain-induced target occlusion scenarios**, never "challenging
#   terrain" - the difficulty is in the target's visibility, not in the driving;
# * procedural relief is **not** claimed to be realistic.  Perlin noise is a
#   procedural representation of spatially correlated natural elevation, and the
#   ``real_world`` field says so rather than pretending the surface is real.
#
# The ``factor`` field is what makes the set more than a list of eight: the
# mechanisms are chosen to span five roughly orthogonal axes.

OCCLUSION_MECHANISMS: dict[str, dict] = {
    "barrier": {
        "label": "lateral occlusion",
        "axis": "occlusion direction", "position": "lateral",
        "real_world": "walls, boulders, shelving, parked machinery",
        "visibility_effect": "the target is behind a finite-width obstacle, so the "
                             "sight line can be restored sideways",
        "physical_effect": "lateral bypass; moving closer does not help",
        "typical_recovery": "side-step around the obstacle's edge",
        "task": "inspection / search",
        "note": "the cleanest Closer is-not More-Observable case",
    },
    "trench": {
        "label": "elongated negative elevation",
        "axis": "elevation", "position": "elongated depression",
        "real_world": "drainage ditches, service trenches, ravines",
        "visibility_effect": "the near lip occludes the target until the robot has "
                             "descended and the far rim is cleared",
        "physical_effect": "the robot's own height changes along the route",
        "typical_recovery": "continue through the trench",
        "task": "infrastructure inspection",
        "note": "solved by forward, forward - a genuinely multi-step case",
    },
    "stairs_up": {
        "label": "discrete height change",
        "axis": "elevation", "position": "discrete steps",
        "real_world": "staircases, industrial platforms, stepped terrain",
        "visibility_effect": "visibility depends on the robot's height, not just "
                             "its horizontal distance",
        "physical_effect": "climbing; horizontal progress alone is insufficient",
        "typical_recovery": "ascend to a higher viewpoint",
        "task": "building inspection",
        "note": "tests action -> future elevation -> visibility",
    },
    "pit": {
        "label": "localised negative elevation",
        "axis": "spatial scale", "position": "local",
        "real_world": "potholes, craters, excavation pits",
        "visibility_effect": "descending lowers the viewpoint and the far wall "
                             "occludes the target",
        "physical_effect": "the viewpoint drops below the surrounding ground",
        "typical_recovery": "back out of the depression",
        "task": "disaster / planetary navigation",
        "note": "solved by back, back - retreat is the correct action",
    },
    "basin": {
        "label": "broad concave elevation field",
        "axis": "spatial scale", "position": "global",
        "real_world": "valleys, mining depressions, large settlement areas",
        "visibility_effect": "visibility varies continuously with elevation across "
                             "a wide region, not at a single lip",
        "physical_effect": "the whole local viewpoint field shifts",
        "typical_recovery": "climb out of the basin",
        "task": "planetary exploration",
        "note": "must not be conflated with pit: local versus broad scale",
    },
    "waves": {
        "label": "periodic visibility transitions",
        "axis": "visibility dynamics", "position": "periodic",
        "real_world": "dunes, wind ripples, rolling natural ground",
        "visibility_effect": "visibility flips repeatedly along the path "
                             "(1 -> 0 -> 1 -> 0 -> 1)",
        "physical_effect": "repeated small pose changes",
        "typical_recovery": "closed-loop re-observation; no permanently correct "
                            "heading exists",
        "task": "outdoor exploration",
        "note": "the direct evidence that visibility is non-monotonic",
    },
    "perlin": {
        "label": "spatially correlated natural relief",
        "axis": "terrain complexity", "position": "smooth",
        "real_world": "natural off-road ground, rock fields, unpaved roads",
        "visibility_effect": "occlusion is distributed over the local terrain "
                             "rather than localised at one obstacle",
        "physical_effect": "continuous pose change along the route",
        "typical_recovery": "terrain-conditioned viewpoint prediction",
        "task": "off-road navigation",
        "note": "procedural relief, not a claim of realism: Perlin noise is a "
                "representation of spatially correlated natural elevation",
    },
    "perlin_rough": {
        "label": "high-frequency natural relief",
        "axis": "terrain complexity", "position": "rough",
        "real_world": "rubble, rock piles, debris fields, rough regolith",
        "visibility_effect": "irregular occlusion with no consistent geometry",
        "physical_effect": "**locomotion uncertainty**: large pose prediction error",
        "typical_recovery": "consequence prediction, not viewpoint geometry",
        "task": "disaster / planetary exploration",
        "note": "the bridge from terrain to future-pose uncertainty",
    },
}

#: The five roughly orthogonal axes the mechanism set is chosen to span.  A
#: mechanism is a *point* on one of these, not a member of a category: pit and
#: basin are both "spatial scale", at opposite ends of it.
MECHANISM_AXES = [
    "occlusion direction",
    "elevation",
    "spatial scale",
    "visibility dynamics",
    "terrain complexity",
]


def mechanism_table() -> list[dict]:
    """The taxonomy as rows, in benchmark order, for tables and figures."""
    return [{"mechanism": key, **OCCLUSION_MECHANISMS[key]}
            for key in BENCHMARK_SCENES if key in OCCLUSION_MECHANISMS]


# --------------------------------------------------------------------------
# Experiment 1: paired-history worlds
# --------------------------------------------------------------------------
# Both worlds share the wall, the lost state (0, 0) and the heading (+x).  They
# differ only in which side of the window the target sits behind, and therefore
# in what the robot saw earlier.  From the shared lost state the target is
# hidden in both, so the current observation is the same; the revealing action
# is a -y strafe in world A and a +y strafe in world B.

HISTORY_LEFT = Scene(
    name="history_left",
    terrain_profile="window_wall",
    target_xy=(5.00, 3.00),
    target_z_absolute=2.00,
    target_height_m=2.00,
    approach_xy=(-2.00, -2.70),
    lost_xy=(0.00, 0.00),
    history_xy=(-2.00, -2.70),
    history_saw_target=True,
    mechanism="target last seen through the window on the +y side",
    notes=("hidden from the shared lost state; only a -y strafe re-reveals it",
           "forward, back and both turns never reveal it"),
)

HISTORY_RIGHT = Scene(
    name="history_right",
    terrain_profile="window_wall",
    target_xy=(5.00, -3.00),
    target_z_absolute=2.00,
    target_height_m=2.00,
    approach_xy=(-2.00, 2.70),
    lost_xy=(0.00, 0.00),
    history_xy=(-2.00, 2.70),
    history_saw_target=True,
    mechanism="target last seen through the window on the -y side",
    notes=("same lost state as history_left, mirrored history",
           "hidden from the shared lost state; only a +y strafe re-reveals it"),
)

HISTORY_PAIRS = {"window": [HISTORY_LEFT, HISTORY_RIGHT]}

SCENES |= {s.name: s for pair in HISTORY_PAIRS.values() for s in pair}


def scene_for(terrain_name: str) -> Scene:
    """Return the scene for a terrain profile.

    Known profiles have a hand-written scene.  Legacy or ad-hoc profiles (the
    sampled height-field generator, a DEM) still need a scene whose
    ``terrain_profile`` matches the requested name, otherwise terrain building
    would silently fall back to a profile that does not exist.
    """
    key = terrain_name.lower()
    if key in SCENES:
        return SCENES[key]
    return replace(GENERIC, name=key, terrain_profile=key)


def eye_offset() -> tuple[float, float, float]:
    """Camera position relative to the robot's base origin."""
    return CAMERA_OFFSET_XYZ
