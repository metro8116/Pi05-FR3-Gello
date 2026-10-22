# 从零配置与复用现有环境

[English](environments.md) · 简体中文

本教程面向 Linux x86_64。机器人电脑与 AutoDL 使用同一仓库，环境按职责隔离。新安装命令只操作本仓库的 `.venv-*`、镜像和 ROS workspace；不会修改原四个项目。当前机器已能运行旧项目时优先显式复用，先不重装驱动或内核。

## 1. 机器人主机的系统前提

推荐新机器使用 Ubuntu 22.04 LTS；已工作的 Ubuntu 20.04 主机可继续使用，ROS Humble 运行在 Ubuntu 22.04 容器内。容器共享主机内核，**安装 Docker 不能替代 Franka 实时控制所需的主机实时内核与调度配置**。按 [Franka 官方安装文档](https://frankarobotics.github.io/docs/installation_linux.html) 与设备支持的 Robot System/libfranka 版本配置实时内核；不要在正在工作的机器上套用不匹配的内核包或固件。

先检查现状：

```bash
uname -a
cat /sys/kernel/realtime 2>/dev/null || true
ip -br addr
ls -l /dev/serial/by-id/
```

主机基础工具（有管理员权限的操作者执行）：

```bash
sudo apt update
sudo apt install -y git curl build-essential pkg-config ffmpeg \
  libgl1 libglib2.0-0 libusb-1.0-0 udev python3-tk openssh-client
```

安装 [Docker Engine 与 Compose plugin](https://docs.docker.com/engine/install/ubuntu/)；检查 `docker info`、`docker compose version`。普通用户需按 Docker 官方方式配置访问权限并重新登录。Docker 权限近似主机管理员权限，不要开放公网 Docker socket。项目显式使用 `unix:///var/run/docker.sock`，避免被其他 Docker context 指向不含设备的引擎。

## 2. FR3 网络与现场模式

当前配置 FR3 地址 `172.16.0.2`，机器人电脑专用有线网卡 `172.16.0.100/24`。在 Ubuntu 网络设置为**实际接机器人**的网口设置静态 IPv4；该专线通常不填默认网关，互联网继续走另一网卡。不要将示例地址应用到错误网卡。

```bash
ip route get 172.16.0.2
ping -c 3 172.16.0.2
```

浏览器打开设备的 Desk 地址（通常 `https://172.16.0.2`），按设备当前软件版本的正常流程解锁关节、解除现场停止、切换 **Execution** 并启用 **FCI**。不绕过机械臂的停止或碰撞保护。`UserStopped` 是设备仍收到停止信号；仅重开采集窗口不会解除它。

在 `configs/local.env` 设置 `FRANKA_IP`（采集用）；部署默认配置 `deploy/configs/deploy.yaml` 的 `robot.ip` 要与之相同。若使用 `deploy/configs/local.yaml` 等私有副本，需通过 `PI05_CONFIG` 选中它。主机持续实时控制，云GPU服务器只接收图像/状态并返回低频目标，无需能直接访问 FR3 专网。

## 3. GELLO、RealSense 与手指

GELLO 用稳定的 `/dev/serial/by-id/...`，不要依赖重启后可能变化的 `/dev/ttyUSB0`。检查自己的设备权限，必要时由管理员添加当前用户到 `dialout`，重新登录。多个串口设备时在 `configs/local.env` 写：

```bash
export GELLO_PORT="/dev/serial/by-id/YOUR_GELLO_DEVICE"
```

仓库 `ros2_ws/src/franka_gello_state_publisher/config/my_gello.yaml` 是当前这副 GELLO 的历史配置，包含 7 个关节的方向、偏置与夹爪行程；另一副设备不能直接照搬。按 GELLO 硬件与上游校准说明测定 `joint_signs`、`best_offsets`、`gripper_range_rad`，先检查数值和方向，再在现场进行小幅测试。参考 [GELLO 上游](https://github.com/wuphilipp/gello_software)。本项目不自动标定、不自动 homing、不自动启用 Dynamixel 力矩。

两台 RealSense 接到足够带宽的 USB 3 接口，避免多个高带宽设备共享不足的 hub。按照 [RealSense SDK Linux 文档](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md) 安装与内核相适配的 SDK/udev 规则，USB 重新插拔后确认权限。项目只取 RGB，不依赖深度流。

当前 camera_1=`233522074047`、camera_2=`405622074065`。位置、朝向、裁剪与采集时尽量一致。这个版本的部署 `config.py` 还锁定本任务的两台序列号和分辨率，避免误换视角；更换相机除修改采集 `TAKE_DATA_CAMERA_SERIALS`、转换器 `--camera-serials`、部署 `cameras.serials`，还需显式更新该校验并重新核对数据/训练契约。不能只交换代码顺序，或直接假定旧模型适用新视角。

夹爪默认 **Move / 0.04 m/s / 总宽 0–0.08 m**，不是按 5 N 力控。Move 到障碍物时实际接触力不由模型的宽度输出限制；更换手指、材料或结构后按其承载条件现场验证，保留已有硬件故障锁直至检查修复。本仓库不清除锁、不执行额外的闭合测试。

## 4. uv 与 Python 环境

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后新开终端，或按照官方输出加载 PATH。项目使用 uv 创建普通隔离 venv，无需 conda，也无需 `uv sync`。按机器角色选择：

```bash
cd /你的路径/Pi05-FR3-Gello
./run.sh setup robot      # 主机 GUI / cameras，Python 3.10
./run.sh setup converter  # 数据转换，Python 3.12 / LeRobot 0.6.1
# AutoDL 上才执行：
./run.sh setup ml         # Python 3.12 / LeRobot 0.6.1
```

依赖详情及验证依据见 [requirements/README.md](../requirements/README_zh.md)。 `setup` 自动选择各 profile 的完整 lock，转换与训练使用 Python 3.12／LeRobot 0.6.1，robot 继续使用 Python 3.10。默认拒绝修改已经存在的同名 `.venv-*`，不会对当前激活的环境安装软件。网络或依赖安装失败后保留环境，不自动删除；修复错误后可执行 `./run.sh setup converter --resume` 继续安装（其他 profile 对应替换名称）。`--resume` 只接受本仓库对应目录内、前缀与 Python 版本正确的真实 venv，拒绝指向其他环境的符号链接；它会安装/更新此目标环境的依赖，不是复用旧环境的命令。

Tkinter 是 OS/Python 的组件。若 uv 管理的 Python `import tkinter` 失败，安装匹配的 Tcl/Tk 与 Python Tk 支持，或用一个确实带 Tk 的 Python 3.10 手动新建 `.venv-robot`，再安装 `requirements/robot.txt`。不要用在 ML 环境安装 GUI 包的方式解决机器人主机的问题。

### 直接复用已工作环境

无需 source activate 或安装。复制模板后在 `configs/local.env` / `configs/autodl.env` 写实际 Python：

```bash
export PI05_ROBOT_PYTHON="/已有/robot环境/bin/python"
export PI05_CONVERTER_PYTHON="/已有/converter环境/bin/python"
export PI05_ML_PYTHON="/已有/训练环境/bin/python"
```

这些解释器从原路径运行，但项目代码来自新仓库。`run.sh` 不调用 pip/uv，只有显式 `setup` 才安装。复用旧环境不会迁移它，也不会改变当前训练进程。旧采集/部署控制会话先正常退出再使用新程序，共享内核锁会防止并发抢占。

## 5. ROS 容器与 VS Code

```bash
./run.sh ros-image   # 网络拉取并编译；首次较久，启动零个机器人节点
./run.sh ros-up      # 只启动 sleep 的空闲容器
./run.sh ros-build   # 编译 collection overlay
./run.sh prepare     # 编译/测试独立 deployment overlay
```

独立容器名 `pi05-fr3-gello-ros2-1`，宿主 `ros2_ws` 挂载 `/workspace/ros2`，仓库挂载 `/workspace/repo`。容器使用主机网络和 USB，保留实时调度能力。镜像不包含数据或模型，也不依赖旧 `gello_software` checkout。`ros-up` 本身不会执行机械臂控制。

ROS underlay 固定 libfranka **0.15.0**、franka_ros2 **v2.0.4**、franka_description **1.0.2** 的 Git commit，见 `docker/Dockerfile`。v2.0.4 内的 `package.xml` 仍声明 2.0.2，不是安装错版本。2026-10-22 已实际构建新镜像，libfranka 和 6 个 Franka ROS 包编译通过；无设备、无网络的临时容器中包查询和动态链接检查通过，详见 [验证记录](verification_zh.md)。尚未在这个新镜像上做硬件实测，Franka firmware 的兼容性要按官方矩阵核对。

`ros-image` 先在主机下载并核验四个固定提交的官方源码包（包括 common 子模块），缓存到 `docker/vendor/`；容器中解包编译，避免 GitHub 全量 clone 的 TLS 中断。已有校验通过的源码包与 Docker 成功层会复用，源码下载及后续 Franka ROS 的 apt 依赖安装设有重试。首个基础 apt 阶段失败时可重新运行统一入口。镜像列出真机控制的 apt 依赖，不再联网读取 rosdep 的 GitHub 索引。缓存不进入 Git，详见 [镜像源码准备说明](../docker/README_zh.md)。

VS Code 安装 Dev Containers 扩展，打开本仓库，再执行 **Dev Containers: Reopen in Container**。`.devcontainer/devcontainer.json` 使用同一 compose service。GUI 在主机执行，VS Code 容器用于 ROS 开发；不需要在容器再安装 Torch。关闭 VS Code 不代表机器人已停止，应先使用程序停止命令。容器内生成的 build/install 可能为 root 所有，后续继续由容器构建，避免手动覆盖原工作区。

## 6. AutoDL 新环境与模型资产

AutoDL 选择支持当前 PyTorch CUDA wheel 的 NVIDIA 驱动/GPU，用 `nvidia-smi` 查看驱动与空闲显存。当前 ML profile 观察版本为 Torch **2.11.0+cu128**；安装系统 CUDA toolkit 通常不是使用该 wheel 的前提，主机驱动必须兼容。batch16 且关闭梯度检查点的内存用量较大，先按训练教程做主动 smoke，不能凭显卡型号保证一定放得下。

已工作的旧 uv 环境、底座与 tokenizer 缓存都可复用，不要升级正在训练的环境。空白机器创建 ML 环境后，需要先联网准备模型和 tokenizer：

```bash
# 在 AutoDL 仓库内；涉及受限模型时先在 HF 接受相应许可并登录。
.venv-ml/bin/hf auth login
.venv-ml/bin/hf download lerobot/pi05_base --local-dir ./models/pi05_base
.venv-ml/bin/hf download google/paligemma-3b-pt-224 \
  --include '*token*' '*vocab*' '*.model' 'config.json' 'special_tokens_map.json' \
  --local-dir ./models/paligemma-tokenizer
```

不要提交 token 或 `models/`。底座目录需有 `config.json` 与单个 `model.safetensors`；当前入口不支持 sharded/adapter-only 资产。若从 HF 缓存复用，指向 `models--lerobot--pi05_base/snapshots/<hash>`，不是外层缓存根目录。实际可用资产以 `./run.sh check-train` 的 tokenizer 和权重头检查为准；如 tokenizer 下载过滤未覆盖其依赖，按错误补齐该 tokenizer 的完整配置/词表文件，无需为了 tokenizer 下载整套视觉语言模型权重。

在 `configs/autodl.env` 设置：

```bash
export PI05_BASE_MODEL="$PI05_REPO_ROOT/models/pi05_base"
export PI05_TOKENIZER="$PI05_REPO_ROOT/models/paligemma-tokenizer"
```

训练/推理读本地资产，默认离线，不会在启动时自动下载。W&B 是可选服务，启用前在 ML 环境执行 `wandb login`，密钥留在机器配置中。

## 7. 安装后检查

```bash
./run.sh doctor                 # 无硬件 inventory
./run.sh collect --check        # 机器人电脑 Python imports；不打开相机
./run.sh check-train --data-only # GPU 主机上有完整数据后再检查
./run.sh deploy --network-check # 服务启动和 SSH 配置完成后；不连接机器人
```

`doctor` 的缺项允许来自另一台机器未安装的 profile。它不会自动修复、重连 FCI、查询密码、开相机或分配大模型；最终按角色检查才能证明相应环境可用。

## 8. 部署过程自动录像

robot profile 增加 imageio-ffmpeg，为新隔离环境提供 FFmpeg。复用旧环境且没有系统 FFmpeg 时，在私有 `configs/local.env` 设置 `PI05_FFMPEG` 指向已有的可执行文件；也可按角色配置安装新 robot 环境。启动录像失败时不会启用模型。录像默认在 `deploy/recordings/`，磁盘需有足够空间；参数与保存状态见部署教程。本次只为现有 Pi05-deploy 的私有配置指定了已经存在的编码器，没有向旧环境安装包。
