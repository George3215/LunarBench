# Trial context
max_steps 是工具调用预算，含拒绝、计划和记录；move_tcp.steps 是本次持续时长倍数。
本地 task2_contact_wall_v1 判据：十块在墙区，4/3/2/1 支撑层，上层跨接两块下层，无机器人接触且低速。
合格边界观测须覆盖至少 1.2 秒仿真时间；不是上游 native benchmark，也不证明连续时间绝不失稳。
释放并退开后，可用当前实测位置、null 姿态及打开夹爪保持观察。finish 不推进时间且不能替代 Python 成功判定。
Python 保存图像、trajectory.jsonl 和状态；sim.npz 为私有证据，不提供在线物体真值。
最新观测提供已记录计划和最近失败。trial 结束后中文复盘 evidence/cause_hypothesis/next_change，
给出 next_trial_plan；host 写 NOTES.md/review.json，下一 trial 恢复同一初始状态并加载经验。此过程不训练权重。
