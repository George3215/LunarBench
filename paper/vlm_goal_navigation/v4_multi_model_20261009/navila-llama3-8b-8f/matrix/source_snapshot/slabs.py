"""Piecewise-linear box terrain for the TASK3 re-observability benchmark.

MuJoCo's height-field collider was measured to stop both vendored locomotion
policies from walking at all in this scene, even though the surface heights agree
with the oracle exactly.  The benchmark therefore defines terrain as an explicit
list of slabs, and every slab becomes one box geom:

* the slab top face is a plane, so flat slabs, ramps and stair treads are exact;
* adjacent slabs of different height produce the risers by themselves;
* a slab may be limited in ``y``, which is what makes "walk around it" a real
  candidate action rather than a cosmetic one.

Because the oracle evaluates :meth:`SlabTerrain.height` from the same slab list
that produces the geoms, the privileged visibility label and the contact surface
cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import math

import numpy as np

RESOLUTION_M = 0.05


@dataclass(frozen=True)
class Slab:
    """One box: a top face that is a plane over ``[x0, x1] x [y0, y1]``."""

    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    thickness_m: float = 1.0
    name: str = "slab"
    rgba: str = "0.45 0.45 0.50 1"

    @property
    def slope(self) -> float:
        return (self.z1 - self.z0) / (self.x1 - self.x0)

    @property
    def pitch(self) -> float:
        return math.atan2(self.z1 - self.z0, self.x1 - self.x0)

    def top_z(self, x: float) -> float:
        return self.z0 + self.slope * (x - self.x0)

    def covers(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def geom_xml(self) -> str:
        """One MuJoCo box whose top face is this slab's plane."""
        length = math.hypot(self.x1 - self.x0, self.z1 - self.z0)
        half_y = (self.y1 - self.y0) / 2.0
        half_t = self.thickness_m / 2.0
        top_cx = (self.x0 + self.x1) / 2.0
        top_cy = (self.y0 + self.y1) / 2.0
        top_cz = (self.z0 + self.z1) / 2.0
        theta = self.pitch
        cx = top_cx - half_t * math.sin(theta)
        cz = top_cz - half_t * math.cos(theta)
        return (f'    <geom name="{self.name}" type="box" '
                f'size="{length / 2.0:.6f} {half_y:.6f} {half_t:.6f}" '
                f'pos="{cx:.6f} {top_cy:.6f} {cz:.6f}" euler="0 {theta:.9f} 0" '
                f'friction="1.0 0.005 0.0001" rgba="{self.rgba}" />\n')


@dataclass
class SlabTerrain:
    """A terrain made of :class:`Slab` boxes, with an analytic height oracle."""

    name: str
    slabs: list[Slab]
    spec: dict
    source: str = "mujoco_slab_kit"
    # Large grid terrains (hundreds of cells) resolve heights by indexing the
    # cell grid instead of testing every slab.  The edges and tops are the box
    # layout itself, so the lookup returns *exactly* the collision surface.
    lookup_x_edges: np.ndarray | None = None
    lookup_y_edges: np.ndarray | None = None
    lookup_tops: np.ndarray | None = None
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    heights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Every box must reach a common floor.  Without this a tall slab (a wall
        # whose top is 3 m above the ground) would be a floating plate with open
        # space underneath, and the rendered camera would happily see the target
        # through that gap while the analytic oracle reported it as blocked.
        self.floor_z = min(min(s.z0, s.z1) for s in self.slabs) - 0.5
        self.slabs = [replace(s, thickness_m=max(s.thickness_m,
                                                 max(s.z0, s.z1) - self.floor_z))
                      for s in self.slabs]
        sx, sy = self.spec["size_m"]
        self.x = np.arange(-sx / 2, sx / 2 + RESOLUTION_M * 0.5, RESOLUTION_M)
        self.y = np.arange(-sy / 2, sy / 2 + RESOLUTION_M * 0.5, RESOLUTION_M)
        xx, yy = np.meshgrid(self.x, self.y)
        self.heights = self._height_from_slabs(xx, yy).astype(np.float32)

    def bottom_z(self) -> float:
        """Lowest box bottom; must be at or below :attr:`floor_z` for every slab."""
        lows = []
        for slab in self.slabs:
            theta = slab.pitch
            half_t = slab.thickness_m / 2.0
            top_cz = (slab.z0 + slab.z1) / 2.0
            cz = top_cz - half_t * math.cos(theta)
            # lowest corner of a box rotated about y
            length = math.hypot(slab.x1 - slab.x0, slab.z1 - slab.z0)
            extent = abs(half_t * math.cos(theta)) + abs((length / 2.0) * math.sin(theta))
            lows.append(cz - extent)
        return float(min(lows))

    def _height_from_slabs(self, px, py):
        """Exact evaluation: the highest slab top covering each point."""
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        out = np.full(np.broadcast(px, py).shape, -1e9, dtype=float)
        for slab in self.slabs:
            inside = (slab.x0 <= px) & (px <= slab.x1) & (slab.y0 <= py) & (py <= slab.y1)
            if not np.any(inside):
                continue
            z = slab.z0 + slab.slope * (px - slab.x0)
            out = np.where(inside, np.maximum(out, z), out)
        if out.ndim == 0:
            return float(out)
        return out

    def height(self, px, py):
        """Highest box top at ``(px, py)``; vectorised when given arrays."""
        if self.lookup_tops is None:
            return self._height_from_slabs(px, py)
        # Cell lookup, not nearest-grid-point: rounding to the closest 0.05 m
        # sample can land on the far side of a cell boundary and report the
        # neighbouring cell's height, which is a whole step of error.
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        xe, ye, tops = self.lookup_x_edges, self.lookup_y_edges, self.lookup_tops
        ix = np.clip(np.searchsorted(xe, px, side="right") - 1, 0, tops.shape[1] - 1)
        iy = np.clip(np.searchsorted(ye, py, side="right") - 1, 0, tops.shape[0] - 1)
        out = tops[iy, ix]
        # A point exactly on a cell edge is shared by both neighbouring cells,
        # and a downward ray in MuJoCo hits whichever top is higher, so the
        # oracle has to take the maximum there as well or `--check` reports a
        # one-quantum mismatch on the sample points that land on an edge.
        left = (ix > 0) & (px == xe[ix])
        below = (iy > 0) & (py == ye[iy])
        out = np.where(left, np.maximum(out, tops[iy, np.maximum(ix - 1, 0)]), out)
        out = np.where(below, np.maximum(out, tops[np.maximum(iy - 1, 0), ix]), out)
        if out.ndim == 0:
            return float(out)
        return out

    def slab_geoms(self) -> str:
        return "".join(slab.geom_xml() for slab in self.slabs)

    def profile(self, xs: np.ndarray, y: float = 0.0) -> np.ndarray:
        return self.height(xs, np.full_like(xs, float(y)))

    def save(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "heightfield.npy", self.heights)
        meta = {**self.spec, "name": self.name, "source": self.source,
                "shape": list(self.heights.shape),
                "x_range_m": [float(self.x[0]), float(self.x[-1])],
                "y_range_m": [float(self.y[0]), float(self.y[-1])],
                "slab_count": len(self.slabs)}
        (directory / "terrain.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (directory / "slabs.json").write_text(
            json.dumps([s.__dict__ for s in self.slabs], indent=2), encoding="utf-8")
        return meta


# --------------------------------------------------------------------------
# Terrain kit
# --------------------------------------------------------------------------

def stairs_up(steps: int = 10, rise: float = 0.15, tread: float = 0.30,
              apron_x: float = 0.0, size_x: float = 12.0, size_y: float = 6.0,
              name: str = "stairs_up") -> SlabTerrain:
    """Ascending staircase in +x: flat apron, ``steps`` treads, long landing."""
    x_min, x_max = -size_x / 2, size_x / 2
    y_half = size_y / 2
    slabs = [Slab(x_min, apron_x, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_apron")]
    for i in range(steps):
        x0 = apron_x + i * tread
        top = (i + 1) * rise
        slabs.append(Slab(x0, x0 + tread, -y_half, y_half, top, top, 1.5, f"{name}_step_{i + 1}"))
    crest = apron_x + steps * tread
    top = steps * rise
    slabs.append(Slab(crest, x_max, -y_half, y_half, top, top, 1.5, f"{name}_landing"))
    spec = {"profile": name, "step_count": steps, "step_rise_m": rise, "step_tread_m": tread,
            "crest_x_m": crest, "landing_height_m": top, "size_m": [size_x, size_y],
            "resolution_m": RESOLUTION_M}
    return SlabTerrain(name, slabs, spec)


def stairs_down(steps: int = 10, rise: float = 0.15, tread: float = 0.30,
                size_x: float = 12.0, size_y: float = 6.0,
                name: str = "stairs_down") -> SlabTerrain:
    """Descending staircase in +x: high landing, ``steps`` treads, low ground."""
    x_min, x_max = -size_x / 2, size_x / 2
    y_half = size_y / 2
    top_height = steps * rise
    crest = x_min + (size_x / 2 - steps * tread)  # where the top landing ends
    crest = -steps * tread
    slabs = [Slab(x_min, crest, -y_half, y_half, top_height, top_height, 1.5, f"{name}_landing")]
    for i in range(steps):
        x0 = crest + i * tread
        z = top_height - (i + 1) * rise
        slabs.append(Slab(x0, x0 + tread, -y_half, y_half, z, z, 1.5, f"{name}_step_{i + 1}"))
    slabs.append(Slab(0.0, x_max, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_ground"))
    spec = {"profile": name, "step_count": steps, "step_rise_m": rise, "step_tread_m": tread,
            "crest_x_m": crest, "landing_height_m": top_height, "size_m": [size_x, size_y],
            "resolution_m": RESOLUTION_M}
    return SlabTerrain(name, slabs, spec)


def _stepped(x0: float, x1: float, z_from: float, z_to: float, n: int,
             y0: float, y1: float, name: str) -> list[Slab]:
    """A stepped transition: the vendored policy is known to handle <=0.15 m risers."""
    dx = (x1 - x0) / n
    slabs = []
    for i in range(n):
        z = z_from + (z_to - z_from) * (i + 1) / n
        slabs.append(Slab(x0 + i * dx, x0 + (i + 1) * dx, y0, y1, z, z, 1.5,
                          f"{name}_{i + 1}"))
    return slabs


def trench(depth: float = 0.7, half_width: float = 1.2, ramp: float = 1.75,
           far_height: float = 0.55, steps_down: int = 5, steps_up: int = 9,
           size_x: float = 12.0, size_y: float = 6.0, name: str = "trench") -> SlabTerrain:
    """A stepped depression the robot can walk down and out the far side.

    The far ground sits above the near ground, so the far rim occludes the target
    from the floor while a robot that commits forward climbs back into view.
    """
    x_min, x_max = -size_x / 2, size_x / 2
    y_half = size_y / 2
    near_end = -half_width - ramp
    far_start = half_width + ramp
    slabs = [Slab(x_min, near_end, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_near")]
    slabs += _stepped(near_end, -half_width, 0.0, -depth, steps_down, -y_half, y_half,
                      f"{name}_down")
    slabs.append(Slab(-half_width, half_width, -y_half, y_half,
                      -depth, -depth, 1.5, f"{name}_floor"))
    slabs += _stepped(half_width, far_start, -depth, far_height, steps_up, -y_half, y_half,
                      f"{name}_up")
    slabs.append(Slab(far_start, x_max, -y_half, y_half,
                      far_height, far_height, 1.5, f"{name}_far"))
    spec = {"profile": name, "depth_m": depth, "half_width_m": half_width, "ramp_m": ramp,
            "far_height_m": far_height, "steps_down": steps_down, "steps_up": steps_up,
            "size_m": [size_x, size_y], "resolution_m": RESOLUTION_M}
    return SlabTerrain(name, slabs, spec)


def pit(depth: float = 0.55, half_width: float = 1.2, ramp: float = 2.0,
        far_height: float = 0.5, steps_down: int = 4,
        size_x: float = 12.0, size_y: float = 6.0, name: str = "pit") -> SlabTerrain:
    """A depression with a stepped near side and an unclimbable far wall.

    Committing forward walks into the wall; the only action that restores the
    sight line is retreating back up the near steps.
    """
    x_min, x_max = -size_x / 2, size_x / 2
    y_half = size_y / 2
    wall_x = half_width
    near_end = -half_width - ramp
    slabs = [Slab(x_min, near_end, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_near")]
    slabs += _stepped(near_end, -half_width, 0.0, -depth, steps_down, -y_half, y_half,
                      f"{name}_down")
    slabs.append(Slab(-half_width, wall_x, -y_half, y_half,
                      -depth, -depth, 1.5, f"{name}_floor"))
    # A vertical jump from -depth up to far_height: the riser is the box side.
    slabs.append(Slab(wall_x, x_max, -y_half, y_half,
                      far_height, far_height, 1.5, f"{name}_far"))
    spec = {"profile": name, "depth_m": depth, "half_width_m": half_width, "ramp_m": ramp,
            "far_height_m": far_height, "steps_down": steps_down,
            "wall_height_m": far_height + depth, "size_m": [size_x, size_y],
            "resolution_m": RESOLUTION_M}
    return SlabTerrain(name, slabs, spec)


def waves(amplitude: float = 0.35, wavelength: float = 6.0, segments: int = 40,
          quantum: float = 0.05, size_x: float = 12.0, size_y: float = 6.0,
          name: str = "waves") -> SlabTerrain:
    """Undulation quantised into a fine staircase of at most ``quantum`` risers.

    The vendored policies are flat-ground and stair walkers, so a continuous ramp
    is not a fair test; quantising keeps the profile non-monotonic while staying
    inside the measured step capability.
    """
    x_min, x_max = -size_x / 2, size_x / 2
    y_half = size_y / 2
    edges = np.linspace(x_min, x_max, segments + 1)
    zs = amplitude * np.sin(2 * math.pi * edges / wavelength)
    zq = np.round(zs / quantum) * quantum
    slabs = [Slab(edges[i], edges[i + 1], -y_half, y_half, float(zq[i]), float(zq[i]),
                  1.5, f"{name}_{i}") for i in range(segments)]
    spec = {"profile": name, "amplitude_m": amplitude, "wavelength_m": wavelength,
            "segments": segments, "step_quantum_m": quantum,
            "size_m": [size_x, size_y], "resolution_m": RESOLUTION_M}
    return SlabTerrain(name, slabs, spec)


def barrier(wall_x: float = 2.0, wall_width: float = 0.4, wall_height: float = 1.5,
            wall_half_y: float = 0.6, size_x: float = 12.0, size_y: float = 6.0,
            name: str = "barrier") -> SlabTerrain:
    """Flat ground plus a wall segment that is finite in y, so it can be walked around."""
    x_min, x_max = -size_x / 2, size_x / 2
    y_half = size_y / 2
    slabs = [
        Slab(x_min, wall_x, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_near"),
        Slab(wall_x, wall_x + wall_width, -y_half, -wall_half_y, 0.0, 0.0, 1.5, f"{name}_gap_low"),
        Slab(wall_x, wall_x + wall_width, -wall_half_y, wall_half_y,
             wall_height, wall_height, 1.5, f"{name}_wall"),
        Slab(wall_x, wall_x + wall_width, wall_half_y, y_half, 0.0, 0.0, 1.5, f"{name}_gap_high"),
        Slab(wall_x + wall_width, x_max, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_far"),
    ]
    spec = {"profile": name, "wall_x_m": wall_x, "wall_width_m": wall_width,
            "wall_height_m": wall_height, "wall_half_y_m": wall_half_y,
            "size_m": [size_x, size_y], "resolution_m": RESOLUTION_M}
    return SlabTerrain(name, slabs, spec)


def window_wall(wall_x: float = 2.0, wall_width: float = 0.15, wall_height: float = 3.0,
                window_half_y: float = 0.8, sill_z: float = 1.0,
                size_x: float = 12.0, size_y: float = 8.0,
                name: str = "window_wall") -> SlabTerrain:
    """A full-width wall pierced by a raised window.

    The window is a *pinhole*: from a fixed vantage the sight line through it
    lands on one side or the other depending on where the robot stands, and the
    raised sill stops the robot from simply walking through.  Two worlds that
    differ only in which side of the window the target sits behind are locally
    indistinguishable but require opposite lateral moves.
    """
    x_min, x_max = -size_x / 2, size_x / 2
    y_half = size_y / 2
    x1 = wall_x + wall_width
    slabs = [
        Slab(x_min, wall_x, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_near"),
        Slab(wall_x, x1, -window_half_y, window_half_y, sill_z, sill_z, 1.5, f"{name}_sill"),
        Slab(wall_x, x1, -y_half, -window_half_y, wall_height, wall_height, 1.5,
             f"{name}_wall_low"),
        Slab(wall_x, x1, window_half_y, y_half, wall_height, wall_height, 1.5,
             f"{name}_wall_high"),
        Slab(x1, x_max, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_far"),
    ]
    spec = {"profile": name, "wall_x_m": wall_x, "wall_width_m": wall_width,
            "wall_height_m": wall_height, "window_half_y_m": window_half_y,
            "sill_z_m": sill_z, "size_m": [size_x, size_y],
            "resolution_m": RESOLUTION_M}
    return SlabTerrain(name, slabs, spec)


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _grad(h, x, y):
    """Perlin's 8 gradient directions selected by the low bits of ``h``."""
    h = np.asarray(h, dtype=np.int64) & 7
    u = np.where(h < 4, x, y)
    v = np.where(h < 4, y, x)
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


def perlin_noise2d(x, y, seed: int = 0):
    """Classic 2D Perlin gradient noise on integer lattice corners.

    Written out rather than pulled from a library so the terrain is reproducible
    from the seed alone on any machine, and so ``numpy`` stays the only dependency.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(256).astype(np.int64)
    perm = np.concatenate([perm, perm])  # avoid a modulo in the hot path

    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = x - xi
    yf = y - yi
    u = _fade(xf)
    v = _fade(yf)

    xi &= 255
    yi &= 255
    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    x1 = _grad(aa, xf, yf)
    x2 = _grad(ba, xf - 1.0, yf)
    y1 = _grad(ab, xf, yf - 1.0)
    y2 = _grad(bb, xf - 1.0, yf - 1.0)
    lo = x1 + u * (x2 - x1)
    hi = y1 + u * (y2 - y1)
    return lo + v * (hi - lo)


def fbm2d(x, y, octaves: int, lacunarity: float = 2.0, gain: float = 0.5,
          seed: int = 0):
    """Fractional Brownian motion: Perlin noise summed over octaves."""
    total = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape, dtype=float)
    amplitude, frequency, norm = 1.0, 1.0, 0.0
    for octave in range(max(1, octaves)):
        total = total + amplitude * perlin_noise2d(x * frequency, y * frequency,
                                                   seed + 101 * octave)
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / max(norm, 1e-9)


def _walkable(heights: np.ndarray, cell_m: float, max_rise: float,
              passes: int = 6) -> np.ndarray:
    """Clamp the field so no neighbour step exceeds ``max_rise``.

    A bluff is fine as geometry but useless as a test track: the vendored policy
    cannot climb more than about 0.15 m, so an unclamped noise field would leave
    the robot stuck against a wall and produce no visibility transitions.

    This is a Lipschitz projection, done as sequential sweeps.  The sweep has to
    be a real loop, not ``z[1:] = clip(z[1:], z[:-1] +/- r)``: numpy materialises
    the right-hand side against the *old* ``z[:-1]``, so the new neighbour
    differences can reach ``2 * max_rise`` instead of being clamped.
    """
    z = heights.copy()
    ny, nx = z.shape
    for _ in range(passes):
        for j in range(ny):
            row = z[j]
            for i in range(1, nx):
                lo, hi = row[i - 1] - max_rise, row[i - 1] + max_rise
                row[i] = lo if row[i] < lo else (hi if row[i] > hi else row[i])
        for i in range(nx):
            for j in range(1, ny):
                lo, hi = z[j - 1, i] - max_rise, z[j - 1, i] + max_rise
                v = z[j, i]
                z[j, i] = lo if v < lo else (hi if v > hi else v)
    return z


def grid(name: str, z_of, size_x: float = 12.0, size_y: float = 6.0,
         cell_m: float = 0.30, quantum: float = 0.05,
         max_rise_m: float = 0.10) -> SlabTerrain:
    """Build a slab terrain from an arbitrary height function on a cell grid.

    Each cell becomes one box whose top is the quantised height at that cell, so
    the collision surface *is* the oracle's surface, and the oracle resolves
    heights by nearest-cell lookup instead of testing every box.
    """
    nx = int(round(size_x / cell_m))
    ny = int(round(size_y / cell_m))
    x_edges = np.linspace(-size_x / 2, size_x / 2, nx + 1)
    y_edges = np.linspace(-size_y / 2, size_y / 2, ny + 1)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(xc, yc)
    raw = np.asarray(z_of(xx, yy), dtype=float)
    stepped = np.round(raw / quantum) * quantum
    stepped = np.round(_walkable(stepped, cell_m, max_rise_m) / quantum) * quantum

    floor = float(stepped.min()) - 0.5
    slabs = []
    for j in range(ny):
        for i in range(nx):
            top = float(stepped[j, i])
            slabs.append(Slab(float(x_edges[i]), float(x_edges[i + 1]),
                              float(y_edges[j]), float(y_edges[j + 1]),
                              top, top, top - floor, f"{name}_{j}_{i}"))
    spec = {"profile": name, "size_m": [size_x, size_y], "cell_m": cell_m,
            "cells": [nx, ny], "step_quantum_m": quantum,
            "max_rise_m": max_rise_m,
            "height_range_m": [round(float(stepped.min()), 3),
                               round(float(stepped.max()), 3)],
            "resolution_m": RESOLUTION_M}
    terrain = SlabTerrain(name, slabs, spec,
                          lookup_x_edges=x_edges, lookup_y_edges=y_edges,
                          lookup_tops=stepped.astype(np.float32))
    return terrain


def perlin(amplitude: float = 0.80, feature_m: float = 3.4, octaves: int = 3,
           cell_m: float = 0.30, quantum: float = 0.05, max_rise_m: float = 0.10,
           seed: int = 5, size_x: float = 12.0, size_y: float = 6.0,
           lateral_blend: float = 0.35, name: str = "perlin") -> SlabTerrain:
    """Rolling ground with Perlin/fBm noise, quantised into walkable steps.

    ``lateral_blend`` controls how much of the 2D noise survives across ``y``.
    Full 2D noise is not walkable here: after clamping to the policy's step limit
    the field becomes a dense field of 0.10-0.15 m ledges, and the robot ends up
    straddling a ledge under its own footprint and stalls (measured: 0.24 m in
    6 s).  Blending toward the ``y = 0`` profile keeps the surface genuinely
    three-dimensional to the eye while holding the lateral variation small
    enough to walk on.  ``lateral_blend = 0`` is a ``y``-invariant profile.

    Defaults were picked by sweeping amplitude, feature size, target height and
    seed and keeping the combination that produced the most visibility
    transitions while staying inside the policy's step limit; an arbitrary seed
    usually gives a single transition.
    """

    def z_of(x, y):
        full = amplitude * fbm2d(x / feature_m, y / feature_m, octaves, seed=seed)
        if lateral_blend >= 1.0:
            return full
        centre = amplitude * fbm2d(x / feature_m, np.zeros_like(x) / feature_m,
                                   octaves, seed=seed)
        return centre + lateral_blend * (full - centre)

    terrain = grid(name, z_of, size_x=size_x, size_y=size_y, cell_m=cell_m,
                   quantum=quantum, max_rise_m=max_rise_m)
    terrain.spec.update({"noise": "perlin_fbm", "octaves": octaves, "seed": seed,
                         "amplitude_m": amplitude, "feature_m": feature_m,
                         "lateral_blend": lateral_blend})
    return terrain


def perlin_rough(amplitude: float = 1.00, feature_m: float = 2.2, octaves: int = 3,
                 cell_m: float = 0.25, quantum: float = 0.05,
                 max_rise_m: float = 0.10, seed: int = 7,
                 size_x: float = 12.0, size_y: float = 6.0,
                 lateral_blend: float = 0.45, name: str = "perlin_rough") -> SlabTerrain:
    """A rougher Perlin field: tighter features, larger amplitude, more lateral relief."""
    return perlin(amplitude=amplitude, feature_m=feature_m, octaves=octaves,
                  cell_m=cell_m, quantum=quantum, max_rise_m=max_rise_m,
                  seed=seed, size_x=size_x, size_y=size_y,
                  lateral_blend=lateral_blend, name=name)


def basin(depth: float = 0.90, floor_half: float = 0.80, steps: int = 6,
          tread: float = 0.40, center_x: float = 1.5, size_x: float = 16.0,
          size_y: float = 6.0, name: str = "basin") -> SlabTerrain:
    """A stepped basin with a flat floor, for putting the *target inside* it.

    Unlike :func:`trench` and :func:`pit`, which hide a target standing beyond a
    far rim, this one hides a target sitting on the floor of the depression.  The
    near rim occludes it while the robot is still outside; walking up to the rim
    and looking down restores line-of-sight, so the mechanism is a grazing angle
    rather than a wall.
    """
    x_min, x_max = -size_x / 2, size_x / 2
    y_half = size_y / 2
    rise = depth / steps
    left_outer = center_x - floor_half - steps * tread
    right_outer = center_x + floor_half + steps * tread
    slabs = [
        Slab(x_min, left_outer, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_left_ground"),
        Slab(right_outer, x_max, -y_half, y_half, 0.0, 0.0, 1.5, f"{name}_right_ground"),
        Slab(center_x - floor_half, center_x + floor_half, -y_half, y_half,
             -depth, -depth, 1.5, f"{name}_floor"),
    ]
    for k in range(steps):
        # The band k treads out from the floor is the (k+1)-th step down from the
        # rim, so the outermost band is shallowest and meets the ground level.
        z = -rise * (k + 1)
        xl0 = left_outer + k * tread
        slabs.append(Slab(xl0, xl0 + tread, -y_half, y_half, z, z, 1.5,
                          f"{name}_left_{k}"))
        zr = -rise * (steps - k)
        xr0 = center_x + floor_half + k * tread
        slabs.append(Slab(xr0, xr0 + tread, -y_half, y_half, zr, zr, 1.5,
                          f"{name}_right_{k}"))
    spec = {"profile": name, "depth_m": depth, "floor_half_m": floor_half,
            "steps": steps, "tread_m": tread, "rise_m": rise,
            "center_x_m": center_x, "floor_z_m": -depth,
            "size_m": [size_x, size_y], "resolution_m": RESOLUTION_M}
    return SlabTerrain(name, slabs, spec)


TERRAIN_KIT = {
    "stairs_up": stairs_up,
    "stairs_down": stairs_down,
    "trench": trench,
    "pit": pit,
    "waves": waves,
    "barrier": barrier,
    "window_wall": window_wall,
    "basin": basin,
    "perlin": perlin,
    "perlin_rough": perlin_rough,
}


def build_terrain(profile: str, **kwargs) -> SlabTerrain:
    if profile not in TERRAIN_KIT:
        raise ValueError(f"unknown slab terrain {profile!r}; known: {sorted(TERRAIN_KIT)}")
    return TERRAIN_KIT[profile](**kwargs)
