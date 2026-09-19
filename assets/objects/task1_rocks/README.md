# TASK1 三维实体石头

123 块原型，由 `tools/prepare_task_rocks.py` 从两套真实来源转换而来，按 id 交替排号：

| 来源 | 数量 | 原始形态 |
| --- | ---: | --- |
| `Asset/environments/lunar/rocks/apollo_rocks/` | 23 | Apollo 样品摄影测量扫描件，三角网格，5–10 万顶点 |
| `Asset/environments/lunar/rocks/lunar_rocks/rocks_s5_r2048/` | 100 | 程序化岩石，四边网格，每个恰好 6146 顶点 |

这里的 `rock*.json` 是**几何修复与简化衍生的资产**，不是新的实物扫描；原始文件不被修改，第三方原始表面的许可边界仍适用。

## 转换做了什么

1. **统一到场地尺度**。两套来源单位互不相同，所以不信任源文件自报的 `metersPerUnit`，直接把每块的**水平最大跨度**归一到 `CANONICAL_EXTENT_M = 2.544`（原来两块原型跨度的均值）——所以每一块都约 2.5 m 跨，`rock.scale` 是唯一的尺寸旋钮：任务当前用 `0.014`，给出约 **3.56 cm** 的石头（全池最长边落在 3.51–4.29 cm），能被 7 cm 的 Piper 夹爪夹住；`0.14` 则给出原来那档 35 cm 左右的石头。
2. **Apollo 需要简化**。用**拓扑保形的边折叠**降到 5000 顶点：每步只并掉一条边的两个端点、删掉边两侧的两个面，邻面自动补位，所以不开孔。
3. **补洞**。源扫描件自带数据缺失留下的洞，按边界边分组、每组扇形封一个中心顶点；这是按构造保流形的，能处理分叉/自触的洞。
4. **贴图只取基色**。Apollo 的 `textures/*.jpg`、lunar 的 usdz 内 `textures/albedo*.png`。

结果：123 块**全部水密**（0 条边界边），123 块有 UV、122 块有配准过的基色贴图。残留 4 条非流形边（`rock10` 1 条、`rock22` 2 条、`rock24` 1 条，整池 219 万条边里的 4 条），来自摄影测量源件的奇点，**故意不修**：去掉只能删面，删了就成洞。

## 文件内容

JSON 中 `points` 为米、右手 Z-up，`indices` 为完整三角面。`uvs` 是**扁平**的 `[u,v,u,v,…]`，每个面角一对、共 `2×len(indices)` 个数；`display.py` 导出给 UE 时会 reshape 成与 `indices` 平行的 `[[u,v],…]`（UE 侧按顶点实例逐个写入）。没有配准贴图的原型不写 `uvs`。

`audit` 记录原始面数、三轴尺寸、体积与闭合性 `closure`（`boundary_edges` / `nonmanifold_edges` / `components`）；`texture` 给出贴图路径与 usdz 的包内条目名（普通图片的 `member` 为 `null`）。

TASK1 同时将此网格送入 MuJoCo 与 UE；MuJoCo 使用 mesh 的原生凸碰撞，显示保留完整凹凸表面。
