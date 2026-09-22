# TASK1 官方实体传感器

> 这份资产现在有第二个使用方：TASK2 空白环境 `tasks/task2_arm/` 把同一份 `d435.obj`
> 装在 **UR12e 的 2F-85 夹爪**上（同样跟着机械臂动）。几何、网格偏移、光学外参与 4 个
> 光学 site 的姿态与下面 TASK1 的口径逐字相同，只有安装位姿不同（TASK1 在 Piper 腕部，
> TASK2 在夹爪 `base` body 上）；TASK2 那边另外定义了两台 MuJoCo 固定相机
> `d435_color` / `d435_depth`，这里只有 site。详见 `tasks/task2_arm/README.md`。
> 下面这一节讲的是 TASK1 的安装，TASK2 的安装不在这一节的范围里。

用户确认布局：**RealSense D435 固定在 PiPER 腕部；Livox Mid-360 固定在 Go2 机身**。照片用于安装布局参考，不能反推出精确安装标定。

机器人固定安装、惯量、碰撞与光学坐标定义在 `../go2_piper/go2_piper.xml`。UE 显示的传感器实体与 MuJoCo 是同一套几何/安装链。RGB-D 严格绑定 `d435_color_optical`；激光射线绑定 `mid360_scan`。拖动观察窗口不会移动它们。

## 官方来源

- Livox 产品：https://www.livoxtech.com/mid-360
- Livox 规格：https://www.livoxtech.com/mid-360/specs
- Livox 官方下载页（STEP、手册、视场模型）：https://www.livoxtech.com/mid-360/downloads
- RealSense D435 产品：https://www.realsenseai.com/products/stereo-depth-camera-d435/
- RealSense 官方描述：https://github.com/realsenseai/realsense-ros/tree/ros2-master/realsense2_description

`SOURCES.json` 记录下载网址、网格校验和与转换矩阵；原始STEP、PDF、官方页面和Xacro保留在工作区 `.local/sensor_sources/`。

CSV中的 librealsense Unity 链接是硬件SDK文档，不是 D435 仿真模型；本次直接使用官方机器人描述与CAD，无需安装Unity、ROS或Isaac。

## D435

- 官方 DAE 已核对 Git blob SHA1：`0ed32368fc5ba6912be530395a72553de9f21d9e`，与此前从上游组合模型复制的文件完全一致。OBJ仅做网格格式转换；Apache-2.0许可见 `INTEL_LICENSE`。
- 尺寸约90×25×25 mm，质量72 g；惯量采用小盒体近似，不把官方Xacro中明确标注不可靠的惯量当实测。
- 腕部安装：link6（含gripper_base），xyz=(-.045,0,.020) m，rpy=(0,-π/2,0)。安装来自 `go2w_sensored_description` 发布URDF，精确支架尺寸未实测。
- 官方名义内部外参：depth/left IR原点=(0,0,0)，right IR=(0,-.050,0)，color=(0,.015,0)，各光学坐标x向右、y向下、z向前。网格本身的(.0043,-.0175,0)偏移保留。
- 官方RGB视场69°×42°、最高1920×1080@30Hz；原生深度视场87°×58°、最高1280×720/最高90Hz（不同模式）。不能把深度FOV套给RGB。
- 当前运行：640×360、RGB水平FOV69°、目标5Hz；深度对齐到RGB视角，名义有效范围.28–10m。输出分辨率/频率为本地计算预算，不冒充完整硬件帧率。
- **UE SceneCapture是实体D435的渲染后端，不是独立场景相机。** 自身外壳不参与其内部光学点的遮挡，外部观察仍显示完整实体。当前是理想几何深度，不模拟双目匹配、IR流、畸变、滚动快门与真实噪声；左右IR site仅定义坐标。

## Mid-360

- 外观使用Livox官网下载的 `mid-360-asm.stp`，经OpenCascade三角化为 `mid360_official.obj`（146383面）；不是第三方社区模型，也不再用几个基本体代替外观。
- 官方外壳尺寸65×65×60 mm，质量265 g；连接器额外突出，转换后的完整CAD包围盒宽约72.74mm。碰撞采用简化外壳盒体，保留独立质量与惯量。
- CAD的+Y向上、+X朝连接器；转换到官方点云坐标+Z向上、+X背离连接器。原点按尺寸图设在底面上方39.5mm。转换矩阵、单位与包围盒记录于 `SOURCES.json`。
- 机身安装：base_link (.270,0,.100) m，pitch=20°。这是可编辑的仿真安装位姿，非照片测量；支架为本地简化几何。
- 官网参数：水平360°、垂直[-7°,52°]，近盲区.1m，40m@10%反射率、70m@80%反射率（100klx），200000点/秒，典型10Hz。
- 当前运行：720射线/帧，目标10Hz，总预算7200射线/秒；40m统一理想量程，低差异非重复采样，保留机器人自身遮挡。**不是厂商真实扫描轨迹或原生点率，不模拟反射率响应、回波强度、测距噪声和逐点运动畸变。** 元数据同时公开官方点率和仿真预算。

官方Livox CAD用于本任务本地仿真；未建立再分发许可，不把它标成自有开源资产。模型转换工具是 `MoonSim/tools/prepare_mid360_asset.py`，只在资产准备时用OpenCascade，运行期不导入CAD库。
