# UR5e EEF contract
单臂 UR5e + Robotiq，不是双 ARX X5；拒绝 left/right 双臂字段。
世界系 +Z 向上、米，目标为绝对 TCP 坐标。姿态用单位 wxyz 或 null（保持实测姿态）。
目标距当前实测 TCP ≤ max_delta_m（上限 0.05 m），最短相对旋转 ≤ max_rotation_rad（上限 0.35 rad）。
超限拒绝，不截短，不推进物理。目标门控不保证 IK 可达、无碰撞或实体位移完全等于目标。
steps 为 1–5，每单位 control_step_seconds 秒；不是 X5 的 25 Hz。一次调用固定同一目标，没有关节直控。
指垫中心=TCP位置+TCP旋转矩阵×pad_offset_tcp；grip_width_m 是净开口，不是闭合百分比。
传输或物理错误终止该次尝试，不重试不确定动作。

move_tcp_delta 的 delta_position 为世界轴下的位移，保持实测姿态，转换为绝对目标后复用同一门控；不是第二条控制连接。

夹爪：grip_width_m=0 表示命令闭合，max_grip_width_m 表示全开；数值减小才是合拢。
例如最大开口约 0.1108 m，继续发送 0.1108 绝不是闭合。接触物体后实测开口不必为 0，也不能据此证明抓住。
策略图像 width/height 和 K 对应本次发送的缩放 JPEG；source_width/source_height 是存盘原图尺寸。
rotation_camera_to_world 将 MuJoCo 相机坐标转为世界坐标：相机 +X 向右、+Y 向上、-Z 向前。
像素 u 向右、v 向下，像素中心从 0 开始；世界观察射线方向为 R @ [(u-cx)/fx, -(v-cy)/fy, -1]。
K 是针孔内参，不包含深度；当前仅发送 RGB，不发送深度。无法确认距离时保留不确定性。
工具结果中的 measured_translation_m、measured_rotation_rad、grip_width_change_m 才是实际变化。
repeated_target_count 和 consecutive_no_motion_calls 是事实反馈，不能把重复保持说成接近或抓取。
