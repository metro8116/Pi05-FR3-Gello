# FR3 / PI05 两机部署

[English](deployment.md) · 简体中文

本机负责相机、GUI、ROS 控制桥和机械臂；AutoDL 负责模型推理。两台机器使用本仓库的同一版本代码。模型和数据集通过各自机器的配置或环境变量指定，不需要另放一份训练工程，也不要把数据、checkpoint、私有 SSH 配置复制进 Git。

默认部署配置为 `deploy/configs/deploy.yaml`，可用 `PI05_CONFIG` 指向自己的配置副本。`run.sh` 使用 `configs/local.env` 与 `configs/autodl.env`；环境文件的创建及 Python 环境选择参见根目录 README。以下命令均从仓库根目录执行，SSH 地址和 SSH 端口使用自己的值。

## 1. 在 AutoDL 启动推理服务

选择一个已经导出的官方 `pretrained_model` checkpoint。服务读取该 checkpoint 的模型配置、原生权重以及保存的 pre/postprocessor 和统计量，不会从训练数据重算统计量。训练和部署直接使用同一个 `fr3_pi05.runtime.load_native_weights_strict`；原生 checkpoint 不走基础模型的键名迁移。

```bash
./run.sh serve --checkpoint checkpoints/SELECTED_CHECKPOINT/pretrained_model \
  --host 127.0.0.1 --port 8000 --device cuda --dtype bfloat16
```

也可以在 AutoDL 环境文件中设置 `PI05_CHECKPOINT`，然后执行 `./run.sh serve`。`--train-project` 默认就是当前仓库根目录，不需要传入另一份训练项目。

先检查本地 checkpoint 和运行环境，可执行：

```bash
./run.sh serve --checkpoint checkpoints/SELECTED_CHECKPOINT/pretrained_model --check-only
```

`--check-only` 不分配 PI05 大模型，但仍检查所选设备和依赖。正式服务默认做一次合成输入预热；它不读取训练数据集，也不连接机械臂。服务只绑定服务器回环地址，通过 SSH 转发访问。加载时会解析 checkpoint 的 `last` 等符号链接，后续训练保存不会把已经加载的服务自动切换到新权重。

仅检查通信链路时可运行：

```bash
./run.sh serve --mock --warmup 0 --device cpu
```

mock metadata 会明确标记 `mock=true`，本机 supervisor 会阻止它启动机器人控制栈。

## 2. 在本机配置 SSH 转发

在本机私有的 `configs/autodl.env` 中设置，保留 `export` 以便监督进程读取这些变量：

```bash
export PI05_SSH_HOST="YOUR_CLOUD_GPU_HOST"
export PI05_SSH_USER="YOUR_SSH_USER"
export PI05_SSH_PORT="YOUR_SSH_PORT"
export PI05_LOCAL_PORT="18000"
export PI05_REMOTE_PORT="8000"
```

把 `YOUR_...` 换成实际配置，先完成 SSH 主机密钥验证和密钥登录。监督进程使用 `BatchMode=yes`、`StrictHostKeyChecking=yes`，不会在后台等待密码或首次主机密钥确认。

正常执行 `./run.sh deploy` 时，supervisor 会管理自己的 SSH 隧道。若希望单独保留隧道，可在另一个本机终端中建立：

```bash
source configs/autodl.env
ssh -N -T -p "$PI05_SSH_PORT" \
  -L "127.0.0.1:18000:127.0.0.1:8000" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  "$PI05_SSH_USER@$PI05_SSH_HOST"
```

随后用 `./run.sh deploy --no-tunnel`，避免重复占用本地端口。本地转发端口必须与部署配置的 `model.port` 一致；服务器端口必须与 `serve --port` 一致。示例中的 `18000` 和 `8000` 是软件默认端口，不是 SSH 登录端口。

只检查隧道及服务 metadata，不启动相机、ROS 或机器人：

```bash
./run.sh deploy --network-check
# 已手动建立隧道时：
./run.sh deploy --network-check --no-tunnel
```

## 3. 准备本机 ROS overlay

按根目录说明准备新 ROS workspace 和新容器。默认容器名为 `pi05-fr3-gello-ros2-1`，本仓库的 `ros2_ws/` 挂载到容器 `/workspace/ros2`。然后运行：

```bash
./run.sh prepare
./run.sh deploy --check
```

`prepare` 构建独立的 `pi05_deploy_overlay` 并运行控制器和夹爪隔离测试，不启动机器人节点。`--check` 检查配置、相机 Python 依赖和 overlay 成功标记，不连接机器人。修改控制器或夹爪源码后必须重新 prepare；缺少 driver binary 或测试成功标记会阻止启动。

独立夹爪包位于 `ros2_ws/src/pi05_gripper_driver`，保留其 LICENSE 和 NOTICE；它使用独立 Stop 路径及状态发布，不替换系统安装的 `franka_gripper` 包。

## 4. GUI 的操作顺序

先结束采集、GELLO 遥操或其他控制会话，再执行：

```bash
./run.sh deploy
# 或使用已经存在的手动隧道：
./run.sh deploy --no-tunnel
```

supervisor 会验证模型 metadata、启动本次 ROS bundle，并等待桥和实际状态就绪后打开 GUI。阻抗控制器初始保持 **inactive**；打开窗口不会自动启用模型控制。

1. 点击“连接并预览”。连接模型、实际状态和两台相机，仍不发送模型动作。
2. 可点击“仅预测（不运动）”。对当前画面和实际状态做一次推理，结果及偏差写入 `client.log`；此操作不提交动作，也不启用控制器。
3. 需要释放物体时，在模型停用、机械臂静止且所有 Move/Stop 已完成的预览状态点击“打开夹爪（80 mm）”。目标为总开口 80 mm，速度 0.04 m/s；机械臂控制器保持 inactive。只有动作成功且新鲜实测开口达到至少 79 mm，GUI 才报告打开完成。
4. 点击“启用模型执行”并确认现场条件，才创建一个新的执行会话。桥检查当前实际姿态、状态新鲜度、模型身份和控制器状态，确认激活后才执行。
5. 点击“停止模型执行”，立即撤销本次会话并独立发送停止请求。窗口和预览保留。再次执行需要等待停止清理完成，并由操作者重新明确启用。

手动开爪、只预测和启用不能相互并发。未明确开启执行时，服务端的推理结果不会自行进入机械臂控制。

### 执行过程自动录像

启用模型执行时，先启动双路真实 RGB 录像、确认两个编码器和首帧可写，再发送 arm。停止模型执行、保护停用、录像出错或关窗时，立即撤销执行并停止接收录像帧，后台保存已有帧。预览、仅预测及单独手动开爪不会启动录像；native Stop 握手等待期间仍保留执行录像。

默认目录为 `deploy/recordings/<time>_<session>_<unique>/`，每次启用新建目录，包含 `camera_1.mp4`、`camera_2.mp4`、`frames.jsonl`、`metadata.json` 和编码日志。旧部署工程使用其 `recordings/`。可用 `PI05_RECORDING_DIR` 覆盖输出根目录；相对路径以部署目录为基准。GUI 显示录像中/保存中/已保存状态、路径、帧数和丢帧数。

保存 640×480 的真实相机 RGB 帧，CPU H.264 编码，标称 30 FPS，不截取 GUI 预览。metadata 保存相机序列号、fps、开始/停止请求/保存完成时间、停止原因、每路帧数和丢帧数；逐帧日志保存源时间戳和编码帧索引。

writer 使用异步有界队列，默认 `recording.queue_size=64`，相机及停止控制线程不等待磁盘。队列满或限时收尾时的丢帧明确记录，不通过重复图像掩盖；MP4 采用标称帧率，存在丢帧时请结合时间戳日志判断实际时长。上一段录像保存期间禁止重新启用。GUI 收到 supervisor 的退出通知时先保存，随后才清理剩余编码器。

需要现成 FFmpeg：依次查 `PI05_FFMPEG`、系统命令、当前 Python 的 `imageio_ffmpeg`、显式 `PI05_CONVERTER_PYTHON` 中的已有 binary，不会下载或安装。新 robot profile 包含 `imageio-ffmpeg`；复用旧环境时可设置 `PI05_FFMPEG`。启动失败不会 arm，运行中录像失败会明确报错并停用模型。此功能只需重开本机部署窗口，不必重建 ROS overlay 或重启 AutoDL 服务。录像与训练数据分开，不自动加入训练集。

## 5. 默认动作接口和门控

以下为仓库当前源码和默认配置值；它们分别约束不同阶段，不能把其中一个超时当作其他超时的替代。

| 项目 | 当前值或行为 |
| --- | --- |
| 任务 | `Put the avocado in the basket.` |
| 状态、动作 | 各 8 维；state 为 7 个实测关节角 rad 加实测夹爪总宽 m，action 为 7 个绝对关节角目标 rad 加目标总宽 m |
| 模型 | 15 Hz，每次返回 50 步；只执行对齐后的最多 6 步前缀，约 0.4 s |
| 本机插值 | 100 Hz；底层控制器 update rate 为 1000 Hz |
| 初始目标差、相邻 waypoint 差 | 最大 0.4 rad、0.15 rad |
| 跟踪误差、关节速度、加速度 | 最大 0.25 rad、0.35 rad/s、0.8 rad/s² |
| 夹爪 | 原生 `Move`，0.04 m/s，总开口 0–0.08 m，deadband 0.002 m |
| Move 接受/执行超时 | 5.0 s；手动开爪总轮询期限为 6.0 s |
| native Stop / 开启抢占握手期限 | 3.0 s |
| 控制器停用确认期限 | 1.2 s |
| 心跳、动作提交超时 | 0.5 s、0.8 s |
| 观测最大年龄 | 1.0 s；不会因此延长动作提交超时 |
| 实测关节、夹爪最大年龄 | 0.2 s、0.5 s |
| 相机 | 固定双相机顺序，640×480 RGB，采集 30 FPS；最大帧龄 0.2 s，双机时间差 0.08 s |

所有 50 步输出仍检查形状、数值和关节界限；不是只检查即将执行的前缀。模型的轻微夹爪越界会记录原始值并裁到物理范围，粗大越界、无效实测状态或不连续关节目标仍被拒绝。

### 夹爪从闭合改为打开时

若旧的闭合 Move 仍在执行，而新鲜模型目标要求打开，桥会走单一 native Stop 路径。它不会同时发送 action cancel，避免两个 Stop 重叠。

GUI 显示“等待夹爪停止，机械臂正在减速保持”。等待期间关节目标经过原有速度/加速度限制逐步减速并保持；这一阶段收到的关节 waypoint 不会在握手结束后回放。新鲜状态、心跳、跟踪误差及其他 watchdog 继续生效。

桥必须同时收到 native Stop 成功和旧 Move 终态，才认为旧目标已结束。此后还要求停止确认之后采集的新鲜夹爪测量，以及仍新鲜的打开目标；不会从异步回调中重放缓存目标。释放开爪后，恢复机械臂轨迹还需要后续新提交。3 秒是停止握手的上限，不是忽略其他故障的宽限期。

## 6. 停止、故障窗口和重启

关闭 GUI 会撤销本次执行，supervisor 随后确认本次 ROS descendants 全部退出并关闭其 SSH 隧道。也可从另一个本机终端执行：

```bash
./run.sh stop-deploy
```

停止按本次 bundle 和 PID/start-time 身份确认，只清理本次拥有的进程。未确认 ROS 完全退出时会保留控制 lease 和恢复 journal 并重试，不把“发出停止请求”当成“已经停止”。

运行中出现 ROS 退出、夹爪致命故障、隧道退出或桥持续失联时，supervisor 仍先停止 ROS 和隧道，但 **保留 GUI** 展示故障与日志路径。故障窗口撤销 token、禁用重连/预测/开爪/启用并停止相机；迟到的 RPC 回复不能恢复旧的 ready 或执行状态。故障窗口不会自动重建 ROS 或重新 arm。

首个有意义的 ROS FATAL/ERROR 及原始来源行写入本次 `session_status.json`，后续 `Connection refused` 或 `Bridge process exiting` 不会遮盖它。只有确认 ROS 和隧道清理后，窗口才显示它们已停止。

保留窗口时控制 lease 和 journal 继续存在，阻止另一份部署或采集占用资源。关闭故障窗口，或执行 `stop-deploy`，才完成退出；然后检查现场状态并重新启动。停止脚本对故障窗口同样生效。

本机 supervisor 还每 0.5 秒探测桥的 state RPC，单次 timeout 为 0.3 秒；持续 2 秒没有有效响应便走本次会话的清理路径。仅 `ready=false` 或普通 fault 且 RPC 仍通，不会被当成进程失联。

## 7. 日志和迁移验证

每次会话写入 `deploy/logs/<session>/`：

- `ros.log`：ROS、控制器和夹爪生命周期；夹爪事件包含目标、实测开口及原始动作结果。
- `ssh.log`：supervisor 管理的 SSH 隧道。
- `client.log`：连接、只预测结果、模型请求及 GUI 错误。
- `session_status.json`：发生运行故障时保存的首因、来源行和清理阶段。

部署复制时仅带入源码、配置和测试；旧 logs、runs、数据、私有 env 和旧验证报告没有作为新仓库内容复制。新部署使用仓库内的共享控制锁实现和训练 native loader；控制锁仍使用 `/tmp/fr3-control-<uid>.lock`，与旧会话保持互斥。

本次部署迁移验证没有启动新的 Docker/ROS 控制会话、部署 SSH 隧道、机器人或 GPU 工作负载。依赖整理另外只读核对过现有 AutoDL 环境。部署 CPU 回归执行了：

```bash
cd deploy
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tests -p test_repository_layout.py -v
```

迁入测试集共 220 项，默认 Python 实际通过 208 项，跳过 12 项：11 项 tiny CPU torch/safetensors 原生加载测试，以及 1 项官方 CPU processor 集成测试，原因是该 Python 缺少对应依赖。另新增 4 项仓库布局测试全部通过，验证新默认路径/容器、共享实现身份、checkpoint 环境变量，以及模拟打包后的桥在没有宿主仓库 `PYTHONPATH` 时能独立显示帮助；初始迁移共 224 项。

加入自动录像后的完整回归：新仓库 252 项，实际通过 239 项、跳过 13 项；现有部署工程 248 项，通过 235 项、跳过 13 项。除上述 12 项 ML 依赖外，默认 Python 还缺少 cv2，因此跳过真实 MP4 解码测试。随后使用已有 robot Python 分别补跑两套 `test_execution_video.py`，各 17 项全部通过，包含真实双相机合成 RGB 的 H.264 MP4 编码及 OpenCV 解码；没有使用真实相机。新增 GUI 保存宽限测试使用无害子进程，确认正常保存前不会向 encoder 发送 TERM。

其中进程生命周期测试只创建本测试拥有的无害 Linux 子进程，使用 fake `ros2`，验证孤儿进程收养和 INT/TERM/KILL 清理；网络测试仅使用本机测试端口。以上结果不代表已经验证真实 GPU 推理、ROS driver 编译或真机运行。准备实际部署时，应在对应 ML 环境补跑被跳过的 CPU 测试，再单独检查现场和真实硬件状态。
