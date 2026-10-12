# terrain_design_v4 Qwen3-VL-8B 试验结果

- 试验数：132；成功：6；成功率：4.55%
- 条件：sensor ∈ {rgb, rgbd} × history ∈ {current, first_current, all_history}
- 终止分布：{'premature_stop': 114, 'out_of_bounds': 11, 'success_visible_near': 6, 'fallen': 1}

## 按协议

| 条件 | 成功 | 总数 | 成功率 |
|---|---:|---:|---:|
| rgb | 1 | 66 | 1.52% |
| rgbd | 5 | 66 | 7.58% |

## 按传感器

| 条件 | 成功 | 总数 | 成功率 |
|---|---:|---:|---:|
| rgb | 1 | 66 | 1.52% |
| rgbd | 5 | 66 | 7.58% |

## 按历史

| 条件 | 成功 | 总数 | 成功率 |
|---|---:|---:|---:|
| all_history | 2 | 44 | 4.55% |
| current | 1 | 44 | 2.27% |
| first_current | 3 | 44 | 6.82% |

## 按条件

| 条件 | 成功 | 总数 | 成功率 |
|---|---:|---:|---:|
| rgb/all_history | 0 | 22 | 0.0% |
| rgb/current | 1 | 22 | 4.55% |
| rgb/first_current | 0 | 22 | 0.0% |
| rgbd/all_history | 2 | 22 | 9.09% |
| rgbd/current | 0 | 22 | 0.0% |
| rgbd/first_current | 3 | 22 | 13.64% |

## 按地形族

| 条件 | 成功 | 总数 | 成功率 |
|---|---:|---:|---:|
| barrier | 0 | 18 | 0.0% |
| basin | 2 | 12 | 16.67% |
| perlin | 1 | 12 | 8.33% |
| perlin_rough | 0 | 18 | 0.0% |
| pit | 0 | 18 | 0.0% |
| slope_up | 0 | 6 | 0.0% |
| stairs_up | 2 | 6 | 33.33% |
| trench | 1 | 6 | 16.67% |
| waves | 0 | 18 | 0.0% |
| window | 0 | 18 | 0.0% |

## 按档位

| 条件 | 成功 | 总数 | 成功率 |
|---|---:|---:|---:|
| complex | 3 | 54 | 5.56% |
| medium | 2 | 42 | 4.76% |
| simple | 1 | 36 | 2.78% |

## 逐回合

| 地形 | mode | horizon | 结果 | 终止 | 自主动作 | 总动作 |
|---|---|---|---|---|---:|---:|
| barrier_a2 | rgb | all_history | fail | premature_stop | 1 | 4 |
| barrier_a2 | rgb | current | fail | premature_stop | 8 | 11 |
| barrier_a2 | rgb | first_current | fail | premature_stop | 5 | 8 |
| barrier_a2 | rgbd | all_history | fail | premature_stop | 4 | 7 |
| barrier_a2 | rgbd | current | fail | premature_stop | 1 | 4 |
| barrier_a2 | rgbd | first_current | fail | premature_stop | 26 | 29 |
| barrier_a1 | rgb | all_history | fail | premature_stop | 1 | 4 |
| barrier_a1 | rgb | current | fail | premature_stop | 8 | 11 |
| barrier_a1 | rgb | first_current | fail | premature_stop | 6 | 9 |
| barrier_a1 | rgbd | all_history | fail | premature_stop | 4 | 7 |
| barrier_a1 | rgbd | current | fail | premature_stop | 1 | 4 |
| barrier_a1 | rgbd | first_current | fail | premature_stop | 19 | 22 |
| barrier_a0 | rgb | all_history | fail | premature_stop | 1 | 4 |
| barrier_a0 | rgb | current | fail | premature_stop | 4 | 7 |
| barrier_a0 | rgb | first_current | fail | premature_stop | 1 | 4 |
| barrier_a0 | rgbd | all_history | fail | premature_stop | 3 | 6 |
| barrier_a0 | rgbd | current | fail | premature_stop | 3 | 6 |
| barrier_a0 | rgbd | first_current | fail | premature_stop | 26 | 29 |
| basin_a2 | rgb | all_history | fail | premature_stop | 1 | 3 |
| basin_a2 | rgb | current | fail | premature_stop | 344 | 346 |
| basin_a2 | rgb | first_current | fail | premature_stop | 2 | 4 |
| basin_a2 | rgbd | all_history | fail | premature_stop | 5 | 7 |
| basin_a2 | rgbd | current | fail | premature_stop | 5 | 7 |
| basin_a2 | rgbd | first_current | fail | out_of_bounds | 27 | 29 |
| basin_a1 | rgb | all_history | fail | premature_stop | 1 | 4 |
| basin_a1 | rgb | current | fail | premature_stop | 15 | 18 |
| basin_a1 | rgb | first_current | fail | premature_stop | 1 | 4 |
| basin_a1 | rgbd | all_history | success | success_visible_near | 10 | 13 |
| basin_a1 | rgbd | current | fail | premature_stop | 6 | 9 |
| basin_a1 | rgbd | first_current | success | success_visible_near | 10 | 13 |
| perlin_a2 | rgb | all_history | fail | premature_stop | 2 | 4 |
| perlin_a2 | rgb | current | fail | premature_stop | 120 | 122 |
| perlin_a2 | rgb | first_current | fail | premature_stop | 3 | 5 |
| perlin_a2 | rgbd | all_history | fail | premature_stop | 3 | 5 |
| perlin_a2 | rgbd | current | fail | premature_stop | 88 | 90 |
| perlin_a2 | rgbd | first_current | success | success_visible_near | 10 | 12 |
| perlin_a1 | rgb | all_history | fail | premature_stop | 1 | 3 |
| perlin_a1 | rgb | current | fail | premature_stop | 84 | 86 |
| perlin_a1 | rgb | first_current | fail | premature_stop | 1 | 3 |
| perlin_a1 | rgbd | all_history | fail | premature_stop | 1 | 3 |
| perlin_a1 | rgbd | current | fail | premature_stop | 7 | 9 |
| perlin_a1 | rgbd | first_current | fail | fallen | 44 | 46 |
| perlin_rough_a2 | rgb | all_history | fail | premature_stop | 1 | 4 |
| perlin_rough_a2 | rgb | current | fail | premature_stop | 38 | 41 |
| perlin_rough_a2 | rgb | first_current | fail | premature_stop | 5 | 8 |
| perlin_rough_a2 | rgbd | all_history | fail | premature_stop | 16 | 19 |
| perlin_rough_a2 | rgbd | current | fail | premature_stop | 37 | 40 |
| perlin_rough_a2 | rgbd | first_current | fail | premature_stop | 1189 | 1192 |
| perlin_rough_a1 | rgb | all_history | fail | premature_stop | 2 | 4 |
| perlin_rough_a1 | rgb | current | fail | premature_stop | 22 | 24 |
| perlin_rough_a1 | rgb | first_current | fail | premature_stop | 5 | 7 |
| perlin_rough_a1 | rgbd | all_history | fail | premature_stop | 3 | 5 |
| perlin_rough_a1 | rgbd | current | fail | premature_stop | 13 | 15 |
| perlin_rough_a1 | rgbd | first_current | fail | out_of_bounds | 41 | 43 |
| perlin_rough_a0 | rgb | all_history | fail | premature_stop | 1 | 3 |
| perlin_rough_a0 | rgb | current | fail | out_of_bounds | 35 | 37 |
| perlin_rough_a0 | rgb | first_current | fail | premature_stop | 77 | 79 |
| perlin_rough_a0 | rgbd | all_history | fail | premature_stop | 2 | 4 |
| perlin_rough_a0 | rgbd | current | fail | premature_stop | 19 | 21 |
| perlin_rough_a0 | rgbd | first_current | fail | premature_stop | 77 | 79 |
| pit_a2 | rgb | all_history | fail | premature_stop | 1 | 6 |
| pit_a2 | rgb | current | fail | premature_stop | 45 | 50 |
| pit_a2 | rgb | first_current | fail | premature_stop | 3 | 8 |
| pit_a2 | rgbd | all_history | fail | premature_stop | 5 | 10 |
| pit_a2 | rgbd | current | fail | premature_stop | 30 | 35 |
| pit_a2 | rgbd | first_current | fail | out_of_bounds | 167 | 172 |
| pit_a1 | rgb | all_history | fail | premature_stop | 1 | 3 |
| pit_a1 | rgb | current | fail | premature_stop | 16 | 18 |
| pit_a1 | rgb | first_current | fail | premature_stop | 2 | 4 |
| pit_a1 | rgbd | all_history | fail | premature_stop | 2 | 4 |
| pit_a1 | rgbd | current | fail | premature_stop | 31 | 33 |
| pit_a1 | rgbd | first_current | fail | out_of_bounds | 25 | 27 |
| pit_a0 | rgb | all_history | fail | premature_stop | 2 | 4 |
| pit_a0 | rgb | current | fail | premature_stop | 43 | 45 |
| pit_a0 | rgb | first_current | fail | premature_stop | 6 | 8 |
| pit_a0 | rgbd | all_history | fail | premature_stop | 3 | 5 |
| pit_a0 | rgbd | current | fail | premature_stop | 109 | 111 |
| pit_a0 | rgbd | first_current | fail | premature_stop | 6 | 8 |
| slope_up_a2 | rgb | all_history | fail | premature_stop | 13 | 16 |
| slope_up_a2 | rgb | current | fail | premature_stop | 35 | 38 |
| slope_up_a2 | rgb | first_current | fail | premature_stop | 11 | 14 |
| slope_up_a2 | rgbd | all_history | fail | premature_stop | 18 | 21 |
| slope_up_a2 | rgbd | current | fail | premature_stop | 2 | 5 |
| slope_up_a2 | rgbd | first_current | fail | out_of_bounds | 79 | 82 |
| stairs_up_a2 | rgb | all_history | fail | premature_stop | 2 | 4 |
| stairs_up_a2 | rgb | current | success | success_visible_near | 9 | 11 |
| stairs_up_a2 | rgb | first_current | fail | premature_stop | 3 | 5 |
| stairs_up_a2 | rgbd | all_history | success | success_visible_near | 10 | 12 |
| stairs_up_a2 | rgbd | current | fail | premature_stop | 20 | 22 |
| stairs_up_a2 | rgbd | first_current | fail | out_of_bounds | 26 | 28 |
| trench_a0 | rgb | all_history | fail | premature_stop | 2 | 4 |
| trench_a0 | rgb | current | fail | premature_stop | 51 | 53 |
| trench_a0 | rgb | first_current | fail | premature_stop | 3 | 5 |
| trench_a0 | rgbd | all_history | fail | premature_stop | 6 | 8 |
| trench_a0 | rgbd | current | fail | premature_stop | 36 | 38 |
| trench_a0 | rgbd | first_current | success | success_visible_near | 7 | 9 |
| waves_a2 | rgb | all_history | fail | premature_stop | 2 | 4 |
| waves_a2 | rgb | current | fail | premature_stop | 3 | 5 |
| waves_a2 | rgb | first_current | fail | premature_stop | 5 | 7 |
| waves_a2 | rgbd | all_history | fail | premature_stop | 3 | 5 |
| waves_a2 | rgbd | current | fail | premature_stop | 16 | 18 |
| waves_a2 | rgbd | first_current | fail | out_of_bounds | 32 | 34 |
| waves_a1 | rgb | all_history | fail | premature_stop | 1 | 4 |
| waves_a1 | rgb | current | fail | premature_stop | 17 | 20 |
| waves_a1 | rgb | first_current | fail | premature_stop | 1 | 4 |
| waves_a1 | rgbd | all_history | fail | premature_stop | 5 | 8 |
| waves_a1 | rgbd | current | fail | premature_stop | 45 | 48 |
| waves_a1 | rgbd | first_current | fail | out_of_bounds | 33 | 36 |
| waves_a0 | rgb | all_history | fail | premature_stop | 1 | 3 |
| waves_a0 | rgb | current | fail | premature_stop | 23 | 25 |
| waves_a0 | rgb | first_current | fail | premature_stop | 11 | 13 |
| waves_a0 | rgbd | all_history | fail | premature_stop | 5 | 7 |
| waves_a0 | rgbd | current | fail | premature_stop | 23 | 25 |
| waves_a0 | rgbd | first_current | fail | out_of_bounds | 29 | 31 |
| window_a2 | rgb | all_history | fail | premature_stop | 2 | 5 |
| window_a2 | rgb | current | fail | premature_stop | 1804 | 1807 |
| window_a2 | rgb | first_current | fail | premature_stop | 1 | 4 |
| window_a2 | rgbd | all_history | fail | premature_stop | 5 | 8 |
| window_a2 | rgbd | current | fail | premature_stop | 1 | 4 |
| window_a2 | rgbd | first_current | fail | premature_stop | 9 | 12 |
| window_a1 | rgb | all_history | fail | premature_stop | 2 | 5 |
| window_a1 | rgb | current | fail | premature_stop | 1793 | 1796 |
| window_a1 | rgb | first_current | fail | premature_stop | 1 | 4 |
| window_a1 | rgbd | all_history | fail | premature_stop | 5 | 8 |
| window_a1 | rgbd | current | fail | premature_stop | 1 | 4 |
| window_a1 | rgbd | first_current | fail | premature_stop | 12 | 15 |
| window_a0 | rgb | all_history | fail | premature_stop | 1 | 6 |
| window_a0 | rgb | current | fail | premature_stop | 21 | 26 |
| window_a0 | rgb | first_current | fail | premature_stop | 1 | 6 |
| window_a0 | rgbd | all_history | fail | premature_stop | 5 | 10 |
| window_a0 | rgbd | current | fail | premature_stop | 2 | 7 |
| window_a0 | rgbd | first_current | fail | out_of_bounds | 33 | 38 |
