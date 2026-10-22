# GELLO / FR3 采集

[English](collection.md) · 简体中文

本仓库的采集界面运行在本机 robot Python 环境，ROS2 Humble、GELLO 发布器及 FR3 控制器运行在独立 `ros2` 容器。启动界面只打开预览，机械臂须在界面中手动启用。

## 配置与准备

- `ros2_ws/src/franka_gello_state_publisher/config/my_gello.yaml` 保留了这台主机此前使用的 GELLO 关节方向、偏移及夹爪行程标定。它是当前硬件的历史校准，其他 GELLO 设备须重新核实和校准。
- `ros2_ws/src/franka_fr3_arm_controllers/config/example_fr3_config.yaml` 中的 `robot_ip` 默认是 `172.16.0.2`。可以修改这份配置，或设置 `FRANKA_IP`（`ROBOT_IP` 也由采集入口接受）。采集 launch 会把它传给 FR3 驱动，不改写 YAML。部署默认配置 `deploy/configs/deploy.yaml` 内的 `robot.ip` 也须与实际地址一致；若使用 `deploy/configs/local.yaml` 等私有副本，需通过 `PI05_CONFIG` 选中它。
- 可以用 `GELLO_PORT=/dev/serial/by-id/设备名` 指定 GELLO；未设置时仅在恰好发现一个 FTDI/OpenRB 设备时自动选择。启动时刷新容器内设备链接并核验设备号，不打开或写入串口；真正的 GELLO 通信在手动启动遥操时开始。
- 默认 `camera_1=233522074047`、`camera_2=405622074065`，可用 `TAKE_DATA_CAMERA_SERIALS=序列号1,序列号2` 覆写。训练与部署必须保持相同顺序与视角；名称本身不代表腕部或外部相机。
- 当前入口固定使用 Franka Hand **Move、0.04m/s、不自动 homing**。Move 只有宽度和速度参数，不提供抓取力设置。不要把当前配置解释成某个已限制的抓取力。
- 控制器沿用 `relative_start=true`：启用时记录 FR3/GELLO 当前角度，之后跟随 GELLO 的变化。它不会自动移动到某个录制起点。

先结束模型部署会话，检查 Franka Desk 的 Execution/FCI 状态，并在现场准备好实体停止开关。采集和部署共用系统控制锁，旧工程与新工程也互斥；不要删除锁文件来抢占控制。

迁移不会导入或清除旧工程的硬件故障锁。若旧工程存在未解除的夹爪故障/检修状态，先完成该状态对应的现场核查；新仓库默认检查 `runtime/gripper_hardware_fault.lock`，可通过 `FR3_HARDWARE_FAULT_LOCK` 指定本机锁文件。该文件存在时拒绝启动。自定义容器内锁路径可用 `FR3_CONTAINER_HARDWARE_FAULT_LOCK`；默认是 `/workspace/repo/runtime/gripper_hardware_fault.lock`。

## 启动与录制

已完成本仓库 robot 环境与 ROS 镜像/工作区配置后，在仓库根目录运行：

```bash
./run.sh collect --output-dir data/raw/avocado_batch01
```

也可直接调用：

```bash
bash capture/start_capture.sh --output-dir data/raw/avocado_batch01
```

默认输出为仓库的 `data/raw/`。新任务或新批次建议使用独立目录，避免把历史记录混入新数据。`TAKE_DATA_OUTPUT_DIR` 与 `--output-dir` 都能配置输出路径。任务默认是 `Put the avocado in the basket.`；可在启动前设置 `TAKE_DATA_TASK`，或在界面中修改。

GUI 中依次操作：

1. 确认两路相机图像、顺序与任务文字。点击“启动遥操作”，等待控制器与真实测量就绪。
2. 摆好起始场景，点击“开始录制”。任务文字为空、遥测缺失或过期时会拒绝录制。
3. 完成任务，保留实际开爪、放下与收手过程；点击“停止录制”，等待文件保存完成。
4. 每段 outcome 默认是 `success`，下一段会重新默认 `success`。失败或人为中断的段在停止录制前改为 `failure` 或 `aborted`。默认标签不能代替操作员对真实完成情况的判断。
5. 继续下一段，或结束采集。

**停止录制只停止保存，不停止遥操作。** 结束时停止录制、点击“停止遥操作”，再关闭 GUI。独立停止控制也可运行：

```bash
bash capture/stop_capture.sh
```

停止入口只读取本仓库 `runs/capture/session.json`，核验目标容器并停止匹配 owner 与 session ID 的进程，不按进程名称全局清扫。没有会话状态时只报告，不扫描或杀进程。停止脚本不关闭 GUI；转换前仍需关闭界面，确保文件不再写入。

若上次异常退出留下会话状态，新遥操启动会拒绝继续。先运行停止入口核验旧会话；不要通过删除状态文件绕过检查。

## 保存的数据

每条记录位于输出目录的 `record_NNN/`，包含：

```text
metadata.json                 最终完整性/任务/outcome权威状态
action/action_data_*.json      对齐后的state/action与原始遥测
action/action_data_*.h5        同一份规范数组和诊断字段
video/camera_1_*.mp4           第一相机RGB
video/camera_2_*.mp4           第二相机RGB
video/*_frames/frame_*.jpg     原始保存图像
video/frame_timestamps.json   相机序列、曝光/接收时间和帧号
```

相机采集为 640×480、30FPS，保存目标帧率为 15FPS。两相机未使用硬件触发同步，转换阶段还会检查时序和帧数。

`observation.state=[实测FR3关节角7维, 两指实测位置之和]`，`action=[控制器实际绝对关节目标7维, 最后接受的夹爪目标宽度]`；关节单位 rad、宽度单位 m，数组为 float32 八维。GELLO 关节读数保留作诊断，不冒充机械臂动作目标。

夹爪 Move 的未达到目标或执行诊断，不代表夹住物体或任务成功。任务 outcome 与低层动作结果独立；数据质量或保存故障仍会把根 metadata 标为 `status=error`，即使 outcome 默认 success。错误段保留供排查，默认转换不选它。

物理运动回放在新采集界面中禁用，检查演示过程请打开 MP4。

## 无硬件检查

```bash
bash capture/start_capture.sh --check
python3 tests/test_capture_session.py
```

`--check` 只导入 Python 依赖，不打开相机、调用 Docker 或连接机械臂。依赖缺失时退出并提示，不自动安装 pip 包、不激活旧 conda。实际解释器由 `CAPTURE_PYTHON` 决定，默认是仓库 `.venv-robot/bin/python`。

`tests/test_capture_contract.py` 在依赖齐全时用模拟遥测和真实 JSON/HDF5 写入验证数据契约；没有相机/GUI依赖的环境会明确跳过，不打开硬件。
