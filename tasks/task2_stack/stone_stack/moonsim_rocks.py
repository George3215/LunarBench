"""用 MoonSim 资产库里的真实岩石做砌墙石头。

形状取自 `MoonSim/assets/objects/rocks/*.usdz`（Apollo 采样石头、lunalab 巨石、
spaceport 月岩，共 33 块），**尺寸 / 密度 / 质量沿用 `rock_wall_stones` 的同一套先验**，
所以和 `paper` / `rough` / `natural` 相比只有"石头长什么样"变了。

USD 用 `pxr`（usd-core）读，转成 trimesh 后走和 paper 风格**同一条**流水线：
取凸包 → OBB 对齐（最长轴→X、最短轴→Z）→ 缩放到目标尺寸。凸包是刻意的：MuJoCo 的
mesh 碰撞本来就只用凸包，视觉形状贴着碰撞形状才不会"看着没碰、实际碰了"。

`irregularity` / `subdivisions` 这两个先验对真实石头没有意义，签名里保留只为和其它
风格一致，调用时忽略。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .rocks import FlatStone
from .rock_wall_stones import ROCK_WALL_TEMPLATES, _align_to_target_obb

#: 默认岩石资产目录：MoonSim/assets/objects/rocks
DEFAULT_ROCK_DIR = Path(__file__).resolve().parents[3] / "assets" / "objects" / "rocks"


def load_usdz_mesh(path: Path | str) -> trimesh.Trimesh:
    """把 `.usdz` 里所有三角面读成一个世界坐标下的 trimesh。

    只取 `UsdGeom.Mesh`；多边形面按扇形三角化；每个 prim 的局部到世界变换会应用，
    否则分层组装的资产会全部叠在原点。
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"打不开 USD 资产: {path}")

    chunks: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        usd_mesh = UsdGeom.Mesh(prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if points.size == 0 or indices.size == 0:
            continue
        matrix = np.array(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        )
        points = points @ matrix[:3, :3].T + matrix[:3, 3]

        cursor = 0
        for count in counts:
            face = indices[cursor:cursor + int(count)]
            cursor += int(count)
            for k in range(1, len(face) - 1):        # 扇形三角化，兼容多边形面
                triangles.append((offset + int(face[0]),
                                  offset + int(face[k]),
                                  offset + int(face[k + 1])))
        chunks.append(points)
        offset += len(points)

    if not chunks or not triangles:
        raise ValueError(f"{path} 里没有可用的三角网格")
    return trimesh.Trimesh(vertices=np.vstack(chunks),
                           faces=np.asarray(triangles, dtype=int), process=True)


def obb_fill_ratio(path: Path | str) -> float:
    """凸包体积 / OBB 体积。越接近 1 越"方"，接触面越大，越好堆也越好夹。

    这个比值在 `_align_to_target_obb` 的各向异性缩放下**保持不变**（凸包和 OBB 的体积
    按同一个行列式缩放），所以它衡量的是形状本身，和最终被拉到多大无关。
    """
    mesh = load_usdz_mesh(path)
    extents = np.sort(mesh.extents)[::-1]
    return abs(float(mesh.convex_hull.volume)) / max(float(np.prod(extents)), 1.0e-12)


def pick_rock_files(rock_dir: Path | str, count: int, seed: int) -> list[Path]:
    """确定性地挑 `count` 个**互不相同**的岩石文件（按 seed 打乱后取前 count 个）。

    试过改成"按 `obb_fill_ratio` 从高到低取最方的"，**实测更差**：规划出来的上层支撑
    面积掉到 1e-05 m² 量级、扰动存活从 0.70 s 掉到 0.44 s，执行后 `stacked` 从 8/10
    降到 6/10。原因是 `_align_to_target_obb` 已经把所有石头拉到同一组模板外接尺寸，
    外接盒宽高比被归一化了，剩下的差异是"填得多满"，而填得满的石头更接近方块、更难
    互相咬合。所以这里保持随机挑选。
    """
    files = sorted(Path(rock_dir).glob("*.usdz"))
    if len(files) < count:
        raise ValueError(f"{rock_dir} 只有 {len(files)} 个 .usdz，要 {count} 个")
    order = np.random.default_rng(seed).permutation(len(files))[:count]
    return [files[int(index)] for index in order]


def generate_moonsim_wall_rock(
    name: str,
    seed: int,
    usd_path: Path | str,
    length: float,
    width: float,
    thickness: float,
    density_range: tuple[float, float] = (1800.0, 2700.0),
) -> FlatStone:
    """读一块真实岩石，对齐到目标尺寸，返回可直接进 MuJoCo 的 `FlatStone`。"""
    rng = np.random.default_rng(seed)
    hull = load_usdz_mesh(usd_path).convex_hull
    hull.fix_normals()
    aligned = _align_to_target_obb(hull, np.array([length, width, thickness], dtype=float))

    volume = max(abs(float(aligned.volume)), 1.0e-9)
    density = float(rng.uniform(*density_range))
    shade = float(rng.uniform(0.34, 0.58))
    warmth = float(rng.uniform(-0.035, 0.045))
    rgba = (
        min(0.72, max(0.22, shade + warmth)),
        min(0.70, max(0.22, shade + 0.5 * warmth)),
        min(0.64, max(0.18, shade * rng.uniform(0.78, 0.96))),
        1.0,
    )
    return FlatStone(
        name=name,
        vertices=[tuple(map(float, vertex)) for vertex in aligned.vertices],
        faces=[tuple(map(int, face)) for face in aligned.faces],
        rgba=rgba,
        mass=density * volume,
        length=length,
        width=width,
        thickness=thickness,
    )


def make_moonsim_wall_rocks(
    seed: int = 17,
    count: int = 10,
    irregularity: float = 1.0,
    subdivisions: int = 5,
    rock_dir: Path | str | None = None,
) -> list[FlatStone]:
    """Create wall stones whose shapes come from MoonSim's real rock assets.

    Sizes come from the same priors as the procedural styles
    (`ROCK_WALL_TEMPLATES` first, then the same uniform ranges), so the wall
    layout and the grasp limits stay comparable.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    files = pick_rock_files(rock_dir or DEFAULT_ROCK_DIR, count, seed)

    stones: list[FlatStone] = []
    dim_rng = np.random.default_rng(seed * 6151 + 97)
    for index in range(count):
        if index < len(ROCK_WALL_TEMPLATES):
            dims = ROCK_WALL_TEMPLATES[index]
        else:
            dims = (float(dim_rng.uniform(0.135, 0.195)),
                    float(dim_rng.uniform(0.095, 0.135)),
                    float(dim_rng.uniform(0.062, 0.095)))
        stone_seed = seed * 100_003 + index * 9_973 + 37
        stones.append(
            generate_moonsim_wall_rock(f"moonsim_{index + 1:02d}", stone_seed,
                                       files[index], *dims)
        )
    return stones
