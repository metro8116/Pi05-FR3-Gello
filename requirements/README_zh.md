# 独立依赖 profiles

[English](README.md) · 简体中文

同一代码库使用三个独立 Python 环境。不要把这三个 requirements 合并，也不要在正在训练的旧环境中安装或升级。`robot.txt`、`converter.txt`、`ml.txt` 固定关键依赖，各自配套 `*.lock.txt` 固定完整解析版本。`setup` 优先安装锁文件，再执行依赖一致性检查。

| 文件 | Python | 用途与验证依据 |
| --- | --- | --- |
| `robot.txt` / `robot.lock.txt` | 3.10 | 本机采集、双 RealSense 预览、远程部署客户端。版本来自工作中的 `take_data` 环境。ROS2/C++ 驱动放在独立容器。 |
| `converter.txt` / `converter.lock.txt` | 3.12，Linux x86_64，glibc >=2.28 | schema-2 原始采集转换为 Dataset v3.0。与训练使用同版 LeRobot **0.6.1**；转换采用 CPU PyTorch 和 PyAV。 |
| `ml.txt` / `ml.lock.txt` | 3.12 | AutoDL 训练和推理。沿用已有 LeRobot **0.6.1**、Torch **2.11.0+cu128**、Transformers **5.5.4** 组合。 |

Dataset **v3.0** 是磁盘数据格式，`lerobot==0.6.1` 是 Python 软件包版本。转换和训练现使用同版 LeRobot，但转换只需要 CPU，不要求采集、转换和训练共用一个 Python 环境。

在项目根目录创建全新的隔离环境，例如：

```bash
uv venv --python 3.12 .venv-converter
uv pip install --python .venv-converter/bin/python -r requirements/converter.lock.txt
uv pip check --python .venv-converter/bin/python
CUDA_VISIBLE_DEVICES='' .venv-converter/bin/python -m unittest discover -s tests -p test_converter_cli.py -v
```

上述命令只创建、安装新的隔离环境。robot 和 ML profile 同样必须选择各自的新环境；ML 的官方 CUDA wheel 直链及 SHA256 写在 `ml.txt`，主机 NVIDIA 驱动需兼容 CUDA 12.8。Tkinter、USB/相机权限、Docker、ROS2、实时内核等由主说明中的主机/容器配置负责，不能通过这些 pip 文件代替。

### 转换依赖选择

LeRobot 0.6.1 要求 Python >=3.12 和 NumPy >=2、<2.3。转换 profile 固定 `numpy==2.2.6`、`opencv-python-headless==4.13.0.92` 和 `h5py==3.15.1`。不能使用 Python 3.10 环境安装此转换 profile，也不使用 `--no-deps` 绕过冲突。

转换只需要 CPU。`torch==2.11.0+cpu` 和配套的 `torchvision==0.26.0+cpu` 使用官方、带 SHA256 的 Python 3.12 / Linux x86_64 wheel URL。它们要求 glibc >=2.28，当前 Ubuntu 20.04 的 glibc 2.31 满足要求。Torch/Torchvision 与 ML profile 使用相同软件版本，转换环境采用 CPU 构建。这里没有额外 CUDA wheel 索引，也不引入 NVIDIA CUDA 或 Triton 包。

数据写入、读取采用显式 PyAV 依赖：`datasets==4.8.5`、`pandas==2.3.3`、`pyarrow==26.0.0`、`av==15.1.0`、`jsonlines==4.0.0`、`imageio==2.38.1` 和 `imageio-ffmpeg==0.6.0`。不启用会引入 TorchCodec 的 `[dataset]` / `[training]` extra，也不安装 PI05/Transformers extra、不加载模型。转换与读回时明确选择 PyAV；其他工具如果沿用 LeRobot 默认的 TorchCodec 后端，需要自行改用 PyAV。

2026-10-22 已用 uv 0.10.12 按 Python 3.12、本机 glibc 2.31 平台完整解析 65 个依赖，并通过 `--only-binary :all:` 检查，全部有兼容 wheel。0.6.1 的硬件与可视化功能是可选依赖，这个转换 profile 不引入 `pynput`、`evdev` 或 `rerun-sdk`。已在本机全新临时 Python 3.12 CPU 环境安装这 65 个包，`uv pip check` 通过；合成双相机记录成功导出 v3.0，并完成视频解码与数据读回。正式安装后仍可运行上面的检查与合成测试。

更新关键依赖后，需重新生成并审查 lock，再安装到新环境检查；不要只手改 lock 中的单个版本：

```bash
uv pip compile requirements/converter.txt \
  --python-version 3.12 --python-platform x86_64-manylinux_2_31 \
  --only-binary :all: --no-strip-extras --output-file requirements/converter.lock.txt
```

## 转换流程

默认输入为仓库 `data/raw`，默认输出为新的 `data/lerobot/fr3_avocado_v30`；输出已存在会拒绝覆盖。默认只选完整、成功的 schema-2 episode，全部用于训练，不创建验证集。

```bash
.venv-converter/bin/python tools/convert_to_lerobot_v30.py \
  --input-dir /实际/raw目录 --dry-run --report /实际/检查报告.json

.venv-converter/bin/python tools/convert_to_lerobot_v30.py \
  --input-dir /实际/raw目录 --output-dir /实际/新输出目录
```

检查报告应有 `status=validated`、非空 `selected` 和正数 `total_frames`，才表示有可导出的数据。`--dry-run` 不导入 LeRobot、不创建输出、不连接相机或机器人。可选 `--trim-plan` 使用已审查的裁剪计划；工具不自行推断或删除原始数据。

## 0.6.1 与 Transformers 兼容性

2026-10-22 对现用 AutoDL 环境进行了只读核查：Python 3.12.3，关键版本与 `ml.txt` 一致。Transformers 的 2,271 个 `.py` 文件、LeRobot 的 490 个 `.py` 文件均匹配其已安装 wheel 的 `RECORD` 哈希，没有发现旧式 `transformers_replace` 或手改包源码。因此本 profile **不添加 Transformers 覆盖脚本**。不要把旧教程的 4.x Transformers 替换补丁套到这里的 5.5.4。

仓库自己的权重加载适配仍然必要：首次从 `pi05_base` 初始化允许基础权重键迁移，官方训练 checkpoint 恢复使用原生键严格加载，保持独立的 `lm_head` 与 `embed_tokens`。这段代码属于项目 `src/fr3_pi05`，不修改安装的 LeRobot/Transformers。

转换器已核对 0.6.1 的 `LeRobotDataset.create`、`RGBEncoderConfig`、`add_frame`、`save_episode`、`finalize` 签名，现有动态参数适配与这些接口相符。转换 profile 单独补齐 h5py 和 PyAV 数据栈；实际合成导出与视频读回测试使用本机新建的临时 CPU 环境，避免向正在训练的 AutoDL 环境安装转换依赖。

ML 使用显式 PyAV 数据依赖，没有启用会额外安装 TorchCodec 的 `[training]` extra。Tokenizer 和模型从已有本地目录/缓存离线读取；本项目检查不会自动下载权重。创建新 ML 环境后，先运行项目环境检查和 CPU 小型回归，再由操作者主动运行训练或服务。

部署自动录像额外使用 `imageio-ffmpeg==0.6.0` 提供的 FFmpeg 可执行文件，已加在 robot profile。复用已有且没有该包的机器人环境时，可通过 `PI05_FFMPEG` 指向已经存在的 FFmpeg，不需要为录像修改旧训练环境。这个新增依赖只为新环境显式安装；本次没有向旧 robot 环境安装软件。
