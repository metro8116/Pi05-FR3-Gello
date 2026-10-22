# 本次迁移验证范围

[English](verification.md) · 简体中文

日期：2026-10-22。这个仓库是从当前可用的本地四个工程迁入所需源码，再整理入口、路径、依赖、文档与测试。它没有运行过真实硬件动作；原工程的历史成功试运行不能当作新镜像的安装验收。

初始迁移检查使用已存在的 Python 环境，只读取依赖，没有安装、升级或修改旧环境。后续临时新环境与 ROS 镜像的安装验证见下方更新。CPU 测试使用合成数据、临时目录、fake bridge/相机/设备后端；不会打开真实相机、串口或机械臂。

已验证的内容：

- 新入口帮助、只读 doctor、参数转发、含空格的异地仓库路径与已有环境保护。
- Shell 语法、Python 源码语法、Compose 配置解析和教程链接。
- 采集 headless 数据契约、会话 owner/PID/启动时间过滤、停止失败时保留会话记录、无会话时不扫描进程。
- 转换器 18 种过滤/损坏/无写入回归；真实编码的合成双相机 12 帧 schema-2 记录导出 v3.0，再由 LeRobot 读回与解码。原输入摘要不变。
- 训练官方入口/参数、processor 与 base/native 加载区分、严格权重键/shape/别名、小模型原生保存恢复。初始迁移检查使用过 LeRobot 0.4.4，历史测试记录中相应 0.6.1 专用 remapper 测试跳过。后续转换默认已统一为 LeRobot 0.6.1／Python 3.12，重新安装与导出读回结果见更新记录。
- 部署原有的状态、夹爪交接、watchdog、监督清理、错误保窗、独立 bundle 等 CPU 回归；共享控制锁与 shared native loader 的对象身份检查。

测试中缺 ML/相机依赖或未提供真实数据集的项目显示 skipped。初始迁移与新增录像测试结果见 `docs/verification_results.json`，后续专项验证见下方更新；不要用“全部通过”替代读取跳过原因。

初始迁移时未执行全新环境安装。随后修复安装依赖，并在独立临时环境实际安装 robot 与新版 converter；之后实际重建 ROS 镜像并编译 libfranka、Franka ROS underlay，见下方更新。仍未执行 GPU 完整 PI05 数值训练/续训/推理、部署 SSH 隧道或真机/真实相机测试。依赖整理曾通过只读检查核对现有 AutoDL 的版本与包文件哈希，没有安装、改包或控制训练进程。部署自动录像的完整性由 CPU fake writer 和实际软件 MP4 编解码测试验证；真实 USB 吞吐、磁盘容量、录像负载需下一次现场运行观察。

后续验收按教程依次做 `doctor`、`collect --check`、ROS build/prepare、数据预检查、两步训练/恢复、无机器人网络检查与预览，再由现场操作者启用模型。不要在正在训练的旧环境里用安装/重建命令来做这个验收。

旧工程保持原样的例外：用户随后明确要求把**部署自动双相机录像**同时加入现有 Pi05-deploy，因此该旧部署目录中仅对应功能、配置、测试和说明会更新；旧 GELLO、采集工程、训练工程、现有数据与 checkpoint 不变。

## 转换环境统一到 0.6.1 的更新

已修复 NumPy 依赖冲突，转换默认改为 Python 3.12／LeRobot 0.6.1／CPU Torch 2.11，训练继续使用对应 CUDA 版本。三个 profile 都有完整 lock；ML 的 Torch 索引遮蔽 PyPI 问题改为官方带 SHA256 的 wheel 直链解决。

在全新的临时 venv 中安装 robot 7 项、converter 65 项依赖及本项目，`uv pip check` 全部通过。robot 的 `collect --check` 无硬件导入通过；新版转换器 4 项测试全过，包括 18 种原始数据校验、12 帧双相机合成记录导出 v3.0、解码读回，以及训练数据预检查和 50 步 action chunk 末帧补齐。测试使用完整的牛油果任务/Move 元数据，没有读取或修改实际采集记录。

按用户要求删除了新仓库刚创建的 `.venv-robot`、`.venv-converter`，临时验证环境也已清理，正式环境留给用户从头创建。详细结果见 [converter_environment_061_verification.json](converter_environment_061_verification.json)。

## ROS 镜像下载修复与实际构建

针对 GitHub clone 的 `GnuTLS recv error (-110)`，新增主机源码准备工具、四份固定提交的 SHA256 锁与缓存，包括 libfranka 的 common 子模块。镜像内只离线验证、解包；用明确的真机依赖列表替代 rosdep 在线下载 GitHub 索引。原先已完成的基础 apt/Python 构建缓存得到复用。

2026-10-22 已实际执行 `./run.sh ros-image`，成功生成 `pi05-fr3-gello:humble`，镜像 ID 为 `sha256:fbb3d436f260c0367fefa84fd26d1bb48420c970905c0cead620e58c71b2bf57`。libfranka 0.15.0 与 6 个 Franka ROS 包编译通过。源码工具在系统 Python 3.8 下 17 项测试通过，四份实际源码包离线校验通过；Shell 语法和 Compose 配置解析通过。

随后在无网络、无设备挂载、无特权的临时容器中，检查 8 个 ROS 包路径、FR3 URDF 和 7 个二进制/共享库的动态链接，均通过。再以只读方式挂载新仓库源码，实际编译 4 个自定义包：GELLO publisher、FR3 arm controller、gripper manager、pi05 gripper driver，开启 `BUILD_TESTING`，编译及包查询通过；这次没有运行其运行时测试或任何 ROS 控制节点。临时容器已移除，没有生成宿主的持久 overlay，所以正式使用仍需按教程执行 `ros-up`、`ros-build`，部署还需 `prepare`。

没有启动或连接真实机械臂、相机或串口，也没有修改旧项目、现有 Python 环境、数据、权重或 AutoDL 训练。完整结果见 [ros_image_verification_20261022.json](ros_image_verification_20261022.json)。
