# 常见问题

[English](troubleshooting.md) · 简体中文

**转换报缺 numpy/opencv/h5py，或选中 0 帧。** 当前 shell 的 Python 不一定是转换环境。用 `./run.sh doctor` 检查 `PI05_CONVERTER_PYTHON`，再 `./run.sh validate`；只有环境依赖齐全且选中有效记录才进行转换。不要向正在训练的 ML 环境临时安装转换依赖。

**找不到底座 config.json。** `PI05_BASE_MODEL` 应是同时包含 config.json 和单个 model.safetensors 的真实 snapshot 目录；外层 `models--lerobot--pi05_base` 不是模型。分词器必须已有本地缓存或完整词表/配置。

**续训出现 remapped key 冲突。** 原生 checkpoint 不应用基座权重键迁移。使用本仓库 `./run.sh resume --resume-from ...`，不要对已训练权重调用 pi05_base 初始化流程。完整带上同级 training_state 和 processors；修复不会重写旧 checkpoint，也不会重置 optimizer。

**UserStopped / FCI 不可用。** 在现场核对实体停止、设备解锁与 Desk 模式。按设备流程切 Execution/Activate FCI；编程模式不运行外部关节控制。先正常结束软件会话再用 Programming 调位置，结束后重新开启控制并核对场景。改变初始姿态太多会影响训练分布，软件不会把动作自动变成相对轨迹。

**串口或相机不见。** 检查 USB 供电、带宽、udev 与权限，GELLO 使用 by-id；多台设备指定序列号。不要将 camera_1/2 对调或同时启动两个相机程序。`collect --no-camera` 仅供调试，不生成本双相机训练所需示范。

**Move 夹到物体时 Command failed。** Move 的目标宽度可能小于物体接触宽度，未达到目标与目标未被接受是不同事件。采集代码/转换器保留原始执行诊断，并只允许明确识别的结果；仍需结合视频判断抓持和任务完成。掉物、硬件损坏或真实通信故障不能当作成功。

**想通过减速把夹力设成 0 N。** Move 接口没有力参数。速度、目标宽度与真实接触力不是同一个量；本模型也不预测力。当前使用 Move 0.04 m/s，不声明为 0/5 N。出现手指损坏先按硬件流程停用检查，工具不会自动删除检修锁。

**部署自动停在篮子上。** 检查部署会话 logs/ 中 client 与 ROS 日志的具体首个原因：网络/图像过期、预测偏差、跟踪误差、夹爪 Stop/Move 或设备状态。先区分服务耗时、SSH 往返和本地执行。不要直接取消 watchdog/限位来掩盖错误。当前是串行分段推理，不是 RTC。

**按停止后夹爪动作还未结束。** 软件 Stop 有设备原生停止确认流程；机械臂在切换期间按限速减速保持。等明确确认后才交付后继动作，保留原会话控制锁至清理结束；停止确认失败不能同时再开新窗口。GUI 故障后可保留诊断窗口，退出后再重启。紧急情况使用现场实体停止。

**collect --check 通过，为什么不能运行？** 它只检查 Python imports，不检查摄像头、实时内核、ROS install、FCI 或场景。Docker 新镜像需先 ros-image / ros-up / ros-build，部署还需 prepare。核对角色环境后按教程逐步检查。

**新仓库为什么没数据和权重？** 它是可发布源码仓库，数据、checkpoint、环境、日志与私有配置被忽略。旧资产可保持原路径显式引用；备份与上传由操作者主动执行，教程不会自动移动当前训练目录。

**ros-image 报 GnuTLS recv error (-110)。** 这是下载 GitHub 源码时 TLS 连接中断。新版入口改用固定提交的源码压缩包、SHA256 校验与主机缓存；更新后重新运行 `./run.sh ros-image`，不要重新安装 Python 环境或清空 Docker 缓存。源码与 common 子模块准备均通过后，容器再进行离线解包和编译；apt 仍需正常网络；镜像已用明确的真机依赖列表代替 rosdep 的在线索引。
