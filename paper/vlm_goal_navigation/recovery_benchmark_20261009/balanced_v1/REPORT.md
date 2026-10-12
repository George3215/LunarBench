# Task3 均衡主集 v1

54个合格来源场景筛选为24个；原始验收与冻结实验保持原样。8类各3个，每类简单、中等、复杂各1个。

评级依据类内几何尺度、障碍结构和已记录恢复动作，人工暂定；未使用VLM成功率。台阶和盆地评级置信度较低，初始距离/目标大小及辅助方块会影响难度，后续配对试验应重新校准。恢复动作数是已搜索成功序列长度，不是最短路径。

| 类别 | 简单 | 中等 | 复杂 |
|---|---|---|---|
|台阶|v2/stairs|v2/stairs_down|v2/stairs_up|
|坑洞|v2/pit|design_v3/pit_s23|design_v3/pit_s67|
|盆地|v2/basin|design_v3/basin_s23|design_v3/basin_s41|
|沟渠与沟壑|design_v3/trench_s23|design_v3/gullies_s23|design_v3/gullies_s41|
|方块障碍与迷宫|design_v3/barrier_s23|design_v3/block_maze_s23|design_v3/block_maze_s41|
|随机起伏|v2/perlin|design_v3/perlin_s41|design_v3/perlin_rough_s41|
|连续波浪|design_v3/waves_s23|design_v3/waves_s41|design_v3/waves_s67|
|窗口结构|design_v3/window_s67|design_v3/window_s23|design_v3/window_s41|

hist从16个减至0个；窗口从21个减至3个。坑洞、盆地、沟渠/沟壑共9个，随机起伏与连续波浪各3个。

去重口径：stairs_up/staircase10为几何参数近重复；旧窗口为局部任务同质，不能说全几何或图像完全重复。其余为配额/尺度代表性筛选。逐项理由见excluded.json。

所有24个通过原G1–G4门槛。此结果仅证明可见性恢复验收，不证明能到目标1m内；未在新子集重新运行模型或计算模型成功率。未通过门槛的碎石、平滑坡等不能强行补入，需要后续重新设计并验收。
