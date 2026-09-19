"""把两份月球岩石扫描转成 TASK1 的石头原型（assets/objects/task1_rocks）。

来源（两套混合池，按 id 交替，撒点用 rock.seed 决定位置、`index % len(meshes)` 决定原型）：

- `assets/environments/lunar/rocks/apollo_rocks/`：23 个 Apollo 样品摄影测量模型，
  三角网格、Y 轴朝上、5–10 万顶点。
- `assets/environments/lunar/rocks/lunar_rocks/rocks_s5_r2048/`：100 个程序化岩石，
  四边网格、Z 轴朝上、每个恰好 6146 顶点。

三个刻意的处理，都记在每块石头的 `audit` 里：

1. **统一到场地尺度**。两套来源的单位互不相同（源文件自报的 metersPerUnit 分别是
   0.01 和 1.0，但按它换算 Apollo 样品只有 0.2 mm，显然不是文件的真实意图），
   所以这里不信任源单位，直接把每块石头的**水平最大跨度**归一到 `CANONICAL_EXTENT_M`。
   这个常数取原来两块原型（2.708 m 与 2.380 m）的均值，于是 `rock.scale: 0.14`
   仍然给出和从前一样的石头尺寸（≈0.36 m 跨度）——**任务参数没有为了换模型而改动**。
2. **Apollo 需要简化**。5–10 万顶点直接进 MuJoCo 与 UE 都不现实（原型 JSON 会到
   MB 级，凸包也慢）。这里用**拓扑保形的边折叠**简化到 5000 顶点：每步只并掉一条边的
   两个端点、删掉边两侧的两个面，邻面自动补位，所以**不开孔**（实测每块都 0 边界边）。
   不用"包围盒分格、格内取重心"那种做法——它一次并掉多个顶点，会把石头薄处的两面粘在
   一起撕出真正的洞。简化只用于扫描件；lunar 四边网格 6146 顶点，**只做三角化，不简化**。
3. **补上源扫描件自带的洞**。摄影测量件在缺数据处会留洞（`15556-0` 那块源网格有 15 条
   边界边），转换时沿边界环扇形封盖补掉。非流形边（123 块里 2 块各 1 条）**不修**：
   要去掉它只能删面，删了反而成洞，而它渲染和凸包都不受影响。两类毛病都写进每块石头的
   `audit.closure`，`tools/validate_task1.py` 会复查真正上场的那几块。
4. **贴图只取基色**。Apollo 的 `textures/*.jpg`、lunar 的 `textures/albedo.png`。
   lunar 还有 roughness/normal，但没接（见 tasks/task1_collect/README.md）。
   贴图不复制进仓库，只在原型里记下"从哪读"，由 display.py 按需取用——100 块岩石的
   贴图有 ~190 MB，而一次场景只用其中几十块。

用法：

    PYTHONPATH= /home/lry/miniconda3/envs/moonunreal-mujoco/bin/python tools/prepare_task_rocks.py

离线工具，普通 demo 启动不需要跑。需要 OpenUSD（`usd-core`）。
"""
from pathlib import Path
import json
import zipfile
import numpy as np
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/objects/task1_rocks'
APOLLO = ROOT / 'assets/environments/lunar/rocks/apollo_rocks'
LUNAR = ROOT / 'assets/environments/lunar/rocks/lunar_rocks/rocks_s5_r2048'

# 原 rock1/rock2 的水平最大跨度均值：换模型后石头仍是这个尺寸，任务参数不变。
CANONICAL_EXTENT_M = 2.544
DECIMATE_ABOVE = 20000   # 顶点数超过这个值才做聚类简化
DECIMATE_TARGET = 5000


def _stage_mesh(path):
    """取 stage 里顶点最多的那个 Mesh 的 (points, face_vertex_indices, face_vertex_counts, uv, uv_indices)。"""
    stage = Usd.Stage.Open(str(path))
    best = None
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        if best is None or len(points) > len(best[0]):
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
            uv = np.asarray(primvar.Get(), dtype=float) if primvar.HasValue() else None
            # 两种来源的 st 都是 faceVarying 且**未加索引**（GetIndices() 返回空数组而不是
            # None），也就是 UV 已经按面角顺序和 faceVertexIndices 一一对齐。空数组统一成
            # None，下游才会走"UV 与 indices 平行"这条路。
            uv_indices = np.asarray(primvar.GetIndices(), dtype=np.int64) if primvar.HasValue() else None
            if uv_indices is not None and len(uv_indices) == 0:
                uv_indices = None
            best = (points @ m[:3, :3] + m[3, :3],          # USD 用行向量：p_world = p_local @ R + t
                    np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64),
                    np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64),
                    uv, uv_indices,
                    UsdGeom.GetStageUpAxis(stage))
    if best is None:
        raise ValueError(f'{path} 里没有 Mesh')
    return best


def _to_z_up(points, up_axis):
    """源文件可能是 Y 轴朝上（Apollo 扫描件都是）。转到本项目的 Z 轴朝上。"""
    if up_axis == 'Y':
        return np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])
    if up_axis == 'Z':
        return points
    raise ValueError(f'不支持的上轴 {up_axis}')


def _triangulate(points, indices, counts, uv, uv_indices):
    """任意多边形按扇形三角化；UV 按**面角**展开，和三角化后的 indices 一一对应。

    两套来源的 st 都是 faceVarying，所以每个面角一个 UV；展开成与 indices 平行的数组后，
    下游（UE 的顶点实例）就不必再关心 USD 的 index 规则。
    """
    faces = np.split(indices, np.cumsum(counts)[:-1])
    corners = indices if uv_indices is None else uv_indices
    corner_faces = np.split(corners, np.cumsum(counts)[:-1])
    tri, tri_uv = [], []
    for face, corner in zip(faces, corner_faces):
        for k in range(1, len(face) - 1):
            tri.append([face[0], face[k], face[k + 1]])
            if uv is not None:
                tri_uv.append([corner[0], corner[k], corner[k + 1]])
    tri = np.asarray(tri, dtype=np.int64)
    # 展平成 (3×面数, 2)，与 tri.ravel() 一一对应。
    return tri, (uv[np.asarray(tri_uv, dtype=np.int64)].reshape(-1, 2) if uv is not None else None)


def _closure(points, tri):
    """(边界边数, 非流形边数, 连通片数)：写进审计，让"不开孔"这条能被机器复查。

    边界边 = 只被一个面用到的边，也就是洞；闭合曲面上应为 0。非流形边 = 被三个以上面共用
    的边，摄影测量扫描件里偶有（123 块里 2 块各 1 条）；它渲染和凸包都不受影响，但要去掉
    只能删面，删了反而成洞，所以留着并如实记录。
    """
    tri = np.asarray(tri, dtype=np.int64)
    edges = np.sort(tri[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    parent = list(range(len(points)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in unique:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return {'boundary_edges': int((counts == 1).sum()),
            'nonmanifold_edges': int((counts > 2).sum()),
            'components': len({find(i) for i in range(len(points))})}


def _cap_holes(points, tri, uv):
    """把源网格自带的洞封上。返回 (points, tri, uv, 说明)。

    摄影测量扫描件在缺数据的地方会留洞：`15556-0` 那块源网格就有 15 条边界边，边折叠
    原样保留了下来。洞在 0.36 m 的石头上看不见，但"整池没有洞"是个能机器检查的不变量，
    比"123 块里有 3 块例外"值钱。

    做法是**按边界边分组封盖**，不绕环走：把边界边按共享顶点并成若干组，每组新增一个重心
    顶点，组内每条边界边和它连成一个三角形。源网格的洞会在顶点上分叉（两块缺数据区共用
    一点），绕环的写法遇到分叉就得挑方向、就得猜；按边封盖不管分不分叉都成立，而且封完
    一定还是流形——每条边界边正好多出一个面（1 → 2），每个新顶点引出的边也正好被两个面
    共用。重心落在组内顶点之间，封盖不会戳出石头轮廓。

    三角形取 (下一个点, 当前点, 重心)：边界边的走向是面朝外时绕洞的方向，反过来才是朝外的
    封盖。新顶点的 UV 取该组各角 UV 的均值，接缝处会有一点点拉伸。
    """
    tri = np.asarray(tri, dtype=np.int64)
    if len(tri) == 0 or uv is None:
        return points, tri, uv, None
    directed = tri[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    # 只被一个面用到的边就是洞的边界，它在面上的走向唯一确定了封盖该朝哪边。
    boundary = directed[counts[inverse] == 1]
    if len(boundary) == 0:
        return points, tri, uv, None
    parent = {int(v): int(v) for v in boundary.ravel()}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in boundary:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for a, b in boundary:
        groups.setdefault(find(int(a)), []).append((int(a), int(b)))
    corners = tri.ravel()
    new_points, new_tri, new_uv = list(points), list(tri), list(uv)
    for edges in groups.values():
        ring = sorted({v for edge in edges for v in edge})
        centre_uv = uv[np.isin(corners, ring)].mean(axis=0)
        index = len(new_points)
        new_points.append(np.mean([points[v] for v in ring], axis=0))
        for a, b in edges:
            new_tri.append([b, a, index])
            new_uv.extend([uv[corners == b].mean(axis=0), uv[corners == a].mean(axis=0), centre_uv])
    degree = {}
    for a, b in boundary:
        degree[int(a)] = degree.get(int(a), 0) + 1
        degree[int(b)] = degree.get(int(b), 0) + 1
    pinched = sum(1 for count in degree.values() if count > 2)
    note = f'补洞 {len(groups)} 处（{len(boundary)} 条边界边；源扫描件缺数据留下的洞）'
    if pinched:
        # 两处缺数据在一个顶点上相触时，封盖的辐条会被三个面共用——源网格在那个点本来就是
        # 奇异的，要用一个顶点盖住它就只能这样。渲染和 MuJoCo 的凸包都不受影响。
        note += f'；{pinched} 个顶点上两处缺数据相触，封盖在那里留下非流形边'
    return (np.asarray(new_points), np.asarray(new_tri, dtype=np.int64),
            np.asarray(new_uv, dtype=float), note)


def _decimate(points, tri, uv, target=DECIMATE_TARGET):
    """边折叠简化（保拓扑）。返回 (points, tri, uv, 说明)。

    逐步把一条边的两个端点并成一个（放在中点），并删掉这条边两侧的面，其余邻面改指新
    端点。这是标准做法，**不会开孔**：每步只去掉两个面、并掉一个顶点，剩余面自动补位。

    为什么不用更简单的"包围盒分格、格内取重心"：那种做法会一次并掉多个顶点，把石头薄处
    的两面粘在一起，撕出真正的孔洞——实测一块 100k 顶点的 Apollo 扫描件简化到 5k 会留下
    365 条边界边、总长 16 m 的破洞，而原始网格是完整闭合的。边折叠没有这个问题。

    代价：纯 Python 逐边折叠，100k → 5k 一块约 10 秒，只适合离线工具。
    """
    verts = [list(map(float, p)) for p in points]
    faces = [list(map(int, f)) for f in tri]
    order = np.random.default_rng(20260918)          # 固定顺序，转换可复现
    dead_v = np.zeros(len(verts), bool)
    dead_f = np.zeros(len(faces), bool)
    vert_faces = [set() for _ in verts]
    nbr = [set() for _ in verts]
    for fi, f in enumerate(faces):
        for i in range(3):
            vert_faces[f[i]].add(fi)
            nbr[f[i]].add(f[(i + 1) % 3])
            nbr[f[(i + 1) % 3]].add(f[i])

    alive = len(verts)

    def collapse(a, b):
        """把 b 并进 a。不满足 link condition（并完会变成非流形）就返回 False。"""
        shared = [fi for fi in vert_faces[a] & vert_faces[b] if not dead_f[fi]]
        # 闭合曲面上一条边恰好两侧各一个面；出现别的度数说明网格已经不干净，放弃这条边。
        if len(shared) != 2:
            return False
        apex = set()
        for fi in shared:
            apex |= {v for v in faces[fi] if v != a and v != b}
        # link condition：a、b 的共同邻点必须恰好是两个对顶点，否则折叠会把两个面粘成非流形。
        if (nbr[a] & nbr[b]) != apex or len(apex) != 2:
            return False
        for fi in shared:
            dead_f[fi] = True
        verts[a] = [(verts[a][k] + verts[b][k]) / 2 for k in range(3)]
        for fi in list(vert_faces[b]):
            if dead_f[fi]:
                continue
            faces[fi] = [a if v == b else v for v in faces[fi]]
            vert_faces[a].add(fi)
        for u in nbr[b]:
            if u == a:
                continue
            nbr[u].discard(b)
            nbr[u].add(a)
            nbr[a].add(u)
        nbr[a].discard(b)
        nbr[b].clear()
        vert_faces[b].clear()
        dead_v[b] = True
        return True

    while alive > target:
        live = [f for fi, f in enumerate(faces) if not dead_f[fi]]
        edges = np.unique(np.sort(np.asarray(live, dtype=np.int64)[:, [0, 1, 1, 2, 2, 0]]
                                  .reshape(-1, 2), axis=1), axis=0)
        order.shuffle(edges)
        before = alive
        for a, b in edges:
            if alive <= target:
                break
            if dead_v[a] or dead_v[b]:
                continue
            if collapse(int(a), int(b)):
                alive -= 1
        if alive == before:                            # 一轮下来一次都没折成，收手
            break

    keep_v = np.flatnonzero(~dead_v)
    remap = np.full(len(verts), -1, np.int64)
    remap[keep_v] = np.arange(len(keep_v))
    kept_faces = [f for fi, f in enumerate(faces) if not dead_f[fi]]
    new_tri = remap[np.asarray(kept_faces, dtype=np.int64)]
    new_points = np.asarray(verts)[keep_v]
    new_uv = None
    if uv is not None:
        # 折叠之后面角与顶点的对应关系已经乱了，改成**每顶点一个 UV**（该顶点所有面角 UV
        # 的均值），再展开成面角数组。接缝处的 UV 会被拉平，这是简化换来的代价。
        corners = tri.ravel()
        per_vertex = np.column_stack([
            np.bincount(corners, weights=uv[:, axis], minlength=len(verts)) for axis in range(2)])
        seen = np.bincount(corners, minlength=len(verts)).astype(float)
        per_vertex /= np.maximum(seen, 1)[:, None]
        new_uv = per_vertex[new_tri].reshape(-1, 2)     # 与 tri.ravel() 平行，和其它路径一致
    note = (f'边折叠 {len(points)} → {len(new_points)} 顶点（保拓扑，不开孔；'
            f'UV 改为每顶点一个，接缝被拉平）')
    return new_points, new_tri, new_uv, note



def _normalize(points):
    """水平最大跨度归一到 CANONICAL_EXTENT_M，XY 居中、底面 Z=0。"""
    extent = float(np.max(np.ptp(points[:, :2], axis=0)))
    points = points * (CANONICAL_EXTENT_M / extent)
    points[:, 0] -= (points[:, 0].min() + points[:, 0].max()) / 2
    points[:, 1] -= (points[:, 1].min() + points[:, 1].max()) / 2
    points[:, 2] -= points[:, 2].min()
    return points, extent


def _albedo_member(source):
    """usdz 里基色贴图的条目名。

    不能写死 `textures/albedo.png`：导出器给贴图加了与岩石同号的数字后缀，100 个件里
    只有 1 个正好叫 albedo.png，其余是 albedo.001.png、albedo.002.png……写死的话 99 个
    原型取贴图时会 KeyError。每个包里恰好一个 albedo 条目，按前缀取即可。
    """
    with zipfile.ZipFile(source) as archive:
        names = [n for n in archive.namelist() if Path(n).name.lower().startswith('albedo')]
    if not names:
        return None
    return sorted(names, key=len)[0]          # 有 albedo.png 就优先用它


def _texture(source):
    """贴图来源描述：usdz 里的是成员文件，普通文件直接给路径。存相对仓库根的路径。

    没有贴图就返回 None——不编一个假的。Apollo 里有且只有一个目录（12013-11）的贴图是
    `Coordinate-Unregistered`，即没有配准到模型 UV 上，用了反而贴错，所以也当没有。
    """
    if source is None:
        return None
    if source.suffix == '.usdz':
        member = _albedo_member(source)
        return {'path': str(source.relative_to(ROOT)), 'member': member} if member else None
    return {'path': str(source.relative_to(ROOT)), 'member': None}


def _apollo_texture(directory):
    """目录下配准过的贴图；没有就返回 None。"""
    candidates = sorted(p for p in (directory / 'textures').glob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    registered = [p for p in candidates if 'unregistered' not in p.name.lower()]
    return registered[0] if registered else None


def convert(index, mesh_id, source, texture_source):
    points, indices, counts, uv, uv_indices, up_axis = _stage_mesh(source)
    original = {'vertices': int(len(points)), 'triangles': int(np.sum(counts - 2)),
                'up_axis': str(up_axis), 'face_vertex_counts': sorted(set(counts.tolist()))}
    points = _to_z_up(points, up_axis)
    tri, uv = _triangulate(points, indices, counts, uv, uv_indices)
    points, scale_note = _normalize(points)
    steps = [f'水平跨度 {scale_note:.3f} → {CANONICAL_EXTENT_M} m（源单位不可信，按跨度归一）']
    # 补洞放在简化之前：简化看到的是闭合网格，封盖还能被后续折叠抹平、贴合周围。
    points, tri, uv, hole_note = _cap_holes(points, tri, uv)
    if hole_note:
        steps.append(hole_note)
    if len(points) > DECIMATE_ABOVE:
        points, tri, uv, note = _decimate(points, tri, uv)
        steps.append(note)
    if len(uv) == 0:
        uv = None
    # 面朝外（正体积）：UE 按逆时针为正面，绕反了石头会被背面剔除掉。MuJoCo 只取凸包不受
    # 影响，但显示会整块看不见，所以这里统一一次。
    if np.einsum('ij,ij->i', points[tri[:, 0]], np.cross(points[tri[:, 1]], points[tri[:, 2]])).sum() < 0:
        tri = tri[:, ::-1]
        if uv is not None:
            uv = uv.reshape(-1, 3, 2)[:, ::-1].reshape(-1, 2)
        steps.append('面绕向取反（源网格法线朝内）')
    # 点位 4 位小数（0.1 mm）、UV 4 位小数。不修约的话 Python 的完整浮点 repr 会让这批
    # 原型从 ~40 MB 涨到 200 MB，而这些小数位在 0.36 m 的石头上一像素都看不出来。
    points = np.round(points, 4)
    closure = _closure(points, tri)
    record = {
        'points': points.tolist(),
        'indices': tri.ravel().tolist(),
        'uvs': np.round(uv, 4).ravel().tolist() if uv is not None else None,
        'texture': _texture(texture_source),
        'audit': {
            'id': mesh_id, 'source': str(source.relative_to(ROOT)),
            'source_kind': 'apollo_scan' if 'apollo' in str(source) else 'lunar_procedural',
            'original': original,
            'vertices': int(len(points)), 'triangles': int(len(tri)),
            'extent_m': np.ptp(points, axis=0).round(4).tolist(),
            'closure': closure,
            'has_uv': uv is not None, 'has_texture': texture_source is not None,
            'processing': steps,
        },
    }
    return record


def sources():
    """两套来源按 1:1 交替排号，让前 20 号（stock 场景用到的）两套各占一半。"""
    apollo = sorted(p for p in APOLLO.glob('*/*.usd'))
    lunar = sorted(LUNAR.glob('*.usdz'))
    if not apollo or not lunar:
        raise FileNotFoundError(f'岩石来源为空：apollo={APOLLO} lunar={LUNAR}')
    pairs = []
    for i in range(max(len(apollo), len(lunar))):
        if i < len(lunar):
            pairs.append((lunar[i], lunar[i]))
        if i < len(apollo):
            pairs.append((apollo[i], _apollo_texture(apollo[i].parent)))
    return pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for index, (source, texture_source) in enumerate(sources()):
        mesh_id = index + 1
        record = convert(index, mesh_id, source, texture_source)
        (OUT / f'rock{mesh_id}.json').write_text(
            json.dumps(record, separators=(',', ':'), default=float))
        report.append(record['audit'])
        print(f'{"apollo" if "apollo" in str(source) else "lunar "} rock{mesh_id:<4}'
              f' {record["audit"]["original"]["vertices"]:>7} 顶点 → {record["audit"]["vertices"]:>5} 顶点'
              f' {record["audit"]["triangles"]:>6} 三角面  跨度 {record["audit"]["extent_m"][:2]}')
    (OUT / 'index.json').write_text(json.dumps(report, indent=2, default=float))
    print(f'\n共 {len(report)} 块原型 → {OUT}')
    print(f'简化过的：{sum(1 for r in report if len(r["processing"]) > 1)} 块')


if __name__ == '__main__':
    main()
