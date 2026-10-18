"""Small, deterministic terrain generators for TASK3.

The generator is deliberately independent of OmniLRS.  An OmniLRS DEM can be
passed as an ``.npy`` file, while the built-in profiles make the first test
reproducible on a clean MuJoCo install.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


# Ten physical steps for the TASK3 headline scene.  The rise/tread pair is
# chosen to be walkable in principle by a Go2-sized quadruped, not to be a
# decorative ramp: 0.15 m rise on a 0.30 m tread.
STAIRCASE10_STEPS = 10
STAIRCASE10_RISE_M = 0.15
STAIRCASE10_TREAD_M = 0.30


@dataclass(frozen=True)
class Terrain:
    name: str
    x: np.ndarray
    y: np.ndarray
    heights: np.ndarray
    source: str
    spec: dict

    def height(self, px: float, py: float = 0.0) -> float:
        ix = int(np.clip(np.searchsorted(self.x, px), 1, len(self.x) - 1))
        iy = int(np.clip(np.searchsorted(self.y, py), 1, len(self.y) - 1))
        # nearest-neighbour is intentional: it exposes the same sampled DEM
        # to the oracle and to the simulated contact surface.
        ix -= int(abs(self.x[ix - 1] - px) < abs(self.x[ix] - px))
        iy -= int(abs(self.y[iy - 1] - py) < abs(self.y[iy] - py))
        return float(self.heights[iy, ix])

    def save(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "heightfield.npy", self.heights.astype(np.float32))
        meta = {**self.spec, "name": self.name, "source": self.source,
                "shape": list(self.heights.shape), "x_range_m": [float(self.x[0]), float(self.x[-1])],
                "y_range_m": [float(self.y[0]), float(self.y[-1])]}
        (directory / "terrain.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta


def _grid(size_x: float, size_y: float, resolution: float):
    x = np.arange(-size_x / 2, size_x / 2 + resolution * 0.5, resolution)
    y = np.arange(-size_y / 2, size_y / 2 + resolution * 0.5, resolution)
    return x, y, np.meshgrid(x, y)


def generate(name: str = "stairs", seed: int = 7, resolution: float = 0.05,
             size_x: float = 12.0, size_y: float = 6.0,
             step_rise: float | None = None, step_tread: float | None = None,
             step_count: int | None = None) -> Terrain:
    """Generate a flat, crater, stairs, undulating or staircase10 test surface.

    The ``step_*`` overrides apply to ``staircase10`` only; they let the same
    scene be rebuilt with a different step geometry (for example to measure the
    largest rise the locomotion controller can handle).
    """
    name = name.lower()
    rise = STAIRCASE10_RISE_M if step_rise is None else float(step_rise)
    tread = STAIRCASE10_TREAD_M if step_tread is None else float(step_tread)
    steps = STAIRCASE10_STEPS if step_count is None else int(step_count)
    x, y, (xx, yy) = _grid(size_x, size_y, resolution)
    rng = np.random.default_rng(seed)
    if name == "flat":
        z = np.zeros_like(xx)
    elif name == "crater":
        r = np.sqrt((xx + 0.3) ** 2 + (yy * 1.15) ** 2)
        bowl = -0.42 * np.exp(-(r / 1.65) ** 2)
        rim = 0.20 * np.exp(-((r - 1.65) / 0.23) ** 2)
        z = bowl + rim
    elif name == "stairs":
        # A low stairway creates a repeatable terrain-induced lost-target case.
        z = np.clip(np.floor(np.maximum(xx - 1.15, 0.0) / 0.72), 0, 3) * 0.50
        z += 0.015 * np.sin(yy * 3.0)
    elif name == "undulating":
        z = 0.18 * np.sin(xx * 0.85) + 0.10 * np.sin(yy * 1.7)
        z += 0.035 * rng.standard_normal(xx.shape)
        z = np.clip(z, -0.35, 0.35)
    elif name == "staircase10":
        # The TASK3 headline geometry: ten steps of ``rise`` x ``tread``, a flat
        # apron in front and a long top landing.  The landing is deliberately
        # longer than the step block so that a target standing on it can fall
        # behind the stair crest.
        k = np.floor(np.maximum(xx, 0.0) / tread).astype(np.int32) + 1
        z = np.clip(k, 0, steps) * rise
        z = np.where(xx < 0.0, 0.0, z)
        z = np.where(xx >= steps * tread, steps * rise, z)
    else:
        raise ValueError(f"unknown terrain profile: {name}")
    spec = {"profile": name, "seed": seed, "resolution_m": resolution,
            "size_m": [size_x, size_y]}
    if name == "staircase10":
        spec |= {"step_count": steps,
                 "step_rise_m": rise,
                 "step_tread_m": tread,
                 "crest_x_m": steps * tread,
                 "landing_height_m": steps * rise}
    return Terrain(name, x, y, z.astype(np.float32), "mujoco_procedural", spec)


def load_npy(path: Path, resolution: float = 0.05, name: str = "omnilrs_dem") -> Terrain:
    """Load a DEM exported by OmniLRS as a 2-D NumPy array."""
    h = np.asarray(np.load(path), dtype=np.float32)
    if h.ndim != 2 or min(h.shape) < 3:
        raise ValueError("OmniLRS DEM must be a 2-D array with at least 3 samples per axis")
    x = (np.arange(h.shape[1]) - (h.shape[1] - 1) / 2) * resolution
    y = (np.arange(h.shape[0]) - (h.shape[0] - 1) / 2) * resolution
    return Terrain(name, x, y, h, "omnilrs_npy", {"profile": name, "resolution_m": resolution,
                   "source_dem": str(path)})

