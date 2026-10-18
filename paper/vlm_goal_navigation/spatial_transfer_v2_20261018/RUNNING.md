# 当前执行入口

本轮正式目录：`runs/qwen3vl8b` 与 `runs/spatial_interactor8b`。`smoke/` 全部是接口诊断，不进入44回合正式分母。旧 v1 目录已封存。

Qwen GPU 任务已启动，完成后 `tools/run_task3_spatial_queue.py --root paper/vlm_goal_navigation/spatial_transfer_v2_20261018 --push` 自动串行运行 Spatial。队列保存每个模型约4回合的中间记录，结束后生成配对分析、输入审计、带成功标记的轨迹图并推送 GitHub 实验分支。

实时状态见 `QUEUE_STATUS.json`、两模型 `RUN_STATUS.json`；结果见 `REPORT.md`、`metrics.json`、`episodes.csv`。本文件不将启动或烟测记为完整成功。

本机日志：

- `/tmp/task3_qwen_v2_full_20261018.log`
- `/tmp/task3_spatial_v2_queue_20261018.log`
- Spatial 正式启动后：本目录 `spatial_full.log`

检查命令：

```bash
cat paper/vlm_goal_navigation/spatial_transfer_v2_20261018/QUEUE_STATUS.json
tail -n 15 /tmp/task3_qwen_v2_full_20261018.log
```

切勿在这两个任务运行时修改控制器、模型 prompt、解析器或实验源码。若用户要求暂停，在本目录和当前运行目录各创建 `STOP`；驱动在动作边界停止并保留未完成回合。重启时不得直接覆盖未完成回合。

全部原始RGB-D、序列化输入、模型原文和物理轨迹保留本机；GitHub只备份可审阅协议、摘要、结论和图，不等于原始数据异地备份。
