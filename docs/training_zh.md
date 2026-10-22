# FR3 牛油果任务：数据、π0.5 微调与断点恢复

[English](training.md) · 简体中文

本仓库调用 **LeRobot 0.6.1 官方训练器**。数据采样、Accelerate、优化器、学习率调度、checkpoint 保存及恢复仍由官方实现负责；项目只适配 FR3 数据契约、官方 processor 和严格权重加载。

仓库只包含代码、配置与测试，**没有附带原始采集、清洗后的数据集、`pi05_base` 权重或训练 checkpoint**。这些资产需单独准备，也可以直接引用现有路径。本教程中的训练命令由操作者主动执行；整理仓库、环境检查与 `--dry-run` 不会开始训练。

以下命令从本仓库根目录运行。新布局使用 `data/lerobot/fr3_avocado_v30`，输出放在 `outputs/`。采集与转换参见 [依赖与转换说明](../requirements/README_zh.md)，模型训练好后的部署参见 [部署说明](deployment_zh.md)。

## 1. 选择 AutoDL 上的 ML 环境

如果要复用已经成功训练的旧 uv 环境，不需要重新安装，也不需要执行 `setup ml`：

```bash
cd /root/autodl-tmp/Pi05-FR3-Gello
export PI05_ML_PYTHON="/root/autodl-tmp/mujoco_pick_place_pi05/.venv/bin/python"
"$PI05_ML_PYTHON" -c 'import sys, importlib.metadata as m; print(sys.executable); print("LeRobot:", m.version("lerobot"))'
```

也可将 `PI05_ML_PYTHON` 写入未纳入 Git 的 `configs/autodl.env`。`run.sh` 会读取私有环境文件；若其中已有该变量，以文件配置为准。入口使用所选 Python 运行当前仓库的源码，不要求将当前仓库安装进旧环境。

ML 的已用组合是 Python 3.12、LeRobot 0.6.1、Torch 2.11.0+cu128、Transformers 5.5.4。若使用全新的隔离环境，按根目录环境教程创建 `.venv-ml`，不要把 robot 或 converter 的依赖安装进正在训练的旧环境。转换环境也使用 LeRobot 0.6.1，但只安装 CPU 数据处理依赖；训练环境使用 CUDA 与 PI05 模型依赖，两者保持隔离。

保留原 `HF_HOME`、`HF_HUB_CACHE` 与 tokenizer 资产。现有 AutoDL 缓存位置若为以下目录，可以继续使用：

```bash
export HF_HOME="/root/autodl-tmp/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
```

## 2. 单独准备清洗后的完整数据集

默认训练数据路径为：

```text
Pi05-FR3-Gello/data/lerobot/fr3_avocado_v30/
├── meta/       # info、任务、统计量、episode 元信息、转换来源记录
├── data/       # Parquet 标量数据
└── videos/     # 两路视频
```

必须保留整个导出目录，包括 `meta/conversion_manifest.json` 或兼容的来源 manifest、统计量、来源时间戳及裁剪记录。只复制 MP4、Parquet 或 `info.json` 都不够。训练不依赖原采集目录仍在原位置；manifest 中的历史绝对来源路径仅用于追溯。

已有清洗后的 v3.0 数据无需再次清洗或转换。可在本机单独复制到新仓库的数据目录；以下示例拒绝已有目标：

```bash
CLEAN_DATASET="/home/zezhi/桌面/Pi05-train/data/fr3_avocado_v30"
NEW_DATASET="$PWD/data/lerobot/fr3_avocado_v30"
if [[ ! -f "$CLEAN_DATASET/meta/info.json" ]]; then
  echo "完整源数据集不存在，请先检查路径。" >&2
elif [[ -e "$NEW_DATASET" || -L "$NEW_DATASET" ]]; then
  echo "目标已存在，请保留原目录并选择新的数据集名称。" >&2
else
  mkdir -p "$(dirname "$NEW_DATASET")" && cp -a -- "$CLEAN_DATASET" "$NEW_DATASET"
fi
```

从本机向 AutoDL 单独上传数据的示例：把 SSH 占位符换成自己的配置，目标必须是新的空缺路径；不要把新批次混合上传进已有数据集。

```bash
# 在本机执行；CLEAN_DATASET 是上面的完整 v3.0 目录。
ssh -p YOUR_SSH_PORT root@YOUR_AUTODL_HOST \
  "mkdir -p /root/autodl-tmp/Pi05-FR3-Gello/data/lerobot && \
   test ! -e /root/autodl-tmp/Pi05-FR3-Gello/data/lerobot/fr3_avocado_v30 && \
   test ! -L /root/autodl-tmp/Pi05-FR3-Gello/data/lerobot/fr3_avocado_v30" && \
rsync -a --progress -e 'ssh -p YOUR_SSH_PORT' \
  "$CLEAN_DATASET/" \
  root@YOUR_AUTODL_HOST:/root/autodl-tmp/Pi05-FR3-Gello/data/lerobot/fr3_avocado_v30/
```

先确认第一条 SSH 检查成功，再执行上传；中断后只继续向本次新目标传输同一批数据。原数据与模型均不应提交到 Git。若数据已经在 AutoDL 上，例如旧工程的 `data/fr3_avocado_v30`，可以直接把该绝对路径传给 `--dataset`，不必复制。

本任务的接口固定为：

| 输入或输出 | 含义 |
| --- | --- |
| `observation.state`，8 维 | 7 个实测 FR3 关节角，单位 rad；第 8 维是实测夹爪总开口，单位 m。 |
| `action`，8 维 | 7 个绝对 FR3 关节位置目标；第 8 维是最近接受的夹爪目标总开口，单位 m。 |
| `observation.images.camera_1/2` | 固定顺序的两路 640×480 RGB；官方 reader 提供 CHW 浮点图像。 |
| 频率、任务 | 15 FPS；`Put the avocado in the basket.` |
| 夹爪控制 | 与采集一致的 Move 宽度目标；不是力预测，也不把实际夹到物体后的宽度当作闭合目标。 |

模型处理器负责图像缩放、任务 tokenization 和 state/action 的训练统计归一化。状态与动作使用全量训练行的 quantile 统计；保存的 processor 会随 checkpoint 供推理复用。动作块长度为 50，episode 末尾由官方 reader 补齐，不跨 episode 拼接未来动作。

## 3. 指向已有的完整 `pi05_base` snapshot

新训练使用 `lerobot/pi05_base`，不使用旧仿真任务的微调权重。路径应指向同时含有 `config.json` 和单个完整 `model.safetensors` 的具体 snapshot：

```bash
export PI05_BASE_MODEL="/root/autodl-tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/YOUR_EXISTING_SNAPSHOT_HASH"
test -f "$PI05_BASE_MODEL/config.json"
test -f "$PI05_BASE_MODEL/model.safetensors"
```

将 `YOUR_EXISTING_SNAPSHOT_HASH` 换成已下载的实际目录名。不要只填写外层 `models--lerobot--pi05_base`；那里通常只有 `refs/`、`snapshots/` 和 `blobs/`，不是可加载的模型目录。若 `refs/main` 存在，也可以先查看它指向的 hash，再选择对应 snapshot。代码离线读取，不会替你下载缺失的权重。

Tokenizer 默认从原缓存读取 `google/paligemma-3b-pt-224`。使用独立 tokenizer 目录时，在**检查和新训练**命令中都加 `--tokenizer /实际/tokenizer目录`。checkpoint processor 会保存 tokenizer 来源；它不会自动打包整个 tokenizer 缓存。

## 4. 检查数据、环境和配置

在 AutoDL 执行：

```bash
./run.sh check-train \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --base-model "$PI05_BASE_MODEL"
```

检查覆盖数据契约、全量标量行与统计、两路视频的代表帧解码、50 步动作块及末帧补齐、ML API、CUDA/bfloat16、权重头信息及本地 tokenizer。正常结果包含 `status: passed`、`validation: false`、`episode_selection: all`。这一步不会构建完整 π0.5 网络；完整参数与模型结构的严格匹配发生在真正加载模型时。

只检查数据时使用 `./run.sh check-train --dataset /数据目录 --data-only`；它仍需有 PyAV、LeRobot reader 等数据依赖，但不要求底座权重或 CUDA。

查看正式参数而不加载模型、不创建输出目录：

```bash
./run.sh train \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --policy-path "$PI05_BASE_MODEL" \
  --out ./outputs/fr3_avocado_run2 \
  --wandb true --dry-run
```

`--policy-path` 与 `--base-model`、`--dataset` 与 `--data-root`、`--out` 与 `--output-dir` 分别为同义参数。所有相对数据/输出参数以运行命令时的工作目录为准；从其他目录调用时使用绝对路径。

## 5. 先做两步 smoke 和一次恢复检查

这些命令会真正占用 GPU、执行训练并保存完整模型/优化器。由操作者选择空闲 GPU 时主动运行，输出目录必须尚不存在。不要与占用同一张卡的推理服务或另一个训练任务意外叠加；入口不会自动结束其他进程。

```bash
./run.sh train \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --policy-path "$PI05_BASE_MODEL" \
  --out ./outputs/smoke_2steps \
  --steps 2 --batch-size 1 --save-freq 1 --log-freq 1 \
  --gradient-checkpointing false --wandb false --offline
```

确认保存 `000001`、`000002` 后，从第 1 步恢复到另一个新输出目录：

```bash
./run.sh resume \
  --resume-from ./outputs/smoke_2steps/checkpoints/000001 \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --out ./outputs/smoke_resume
```

这里保留 smoke 配置的**总步数 2**，因此只继续第 2 步。恢复读取 checkpoint 的模型、processor、优化器、scheduler、RNG 与官方训练步数，不重新从底座初始化。不要为 smoke resume 传 `--policy-path` 或改成新的总步数。

## 6. 正式训练：20k / batch 16 / 不开梯度检查点 / 无验证集

需要 W&B 时，先在所选 ML 环境执行 `"$PI05_ML_PYTHON" -m wandb login`，或使用已有登录状态。命令不应包含或记录 API key。

```bash
./run.sh train \
  --policy-path "$PI05_BASE_MODEL" \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --out ./outputs/fr3_avocado_run2 \
  --offline \
  --gradient-checkpointing false \
  --batch-size 16 --num-workers 8 \
  --steps 20000 --save-freq 2000 --log-freq 10 \
  --wandb true
```

选择新的实验目录，例如 `fr3_avocado_run2`。新训练发现输出已存在会拒绝覆盖；继续已有实验应使用下一节的 `resume`。不使用 W&B 时传 `--wandb false`。

| 参数 | 本任务设置 |
| --- | --- |
| 总 optimizer steps / batch size | 20,000 / 16 |
| DataLoader workers / seed | 8 / 1000 |
| 保存 / 日志间隔 | 2,000 / 10 步 |
| dtype / gradient checkpointing | bfloat16 / false |
| freeze vision encoder / train expert only | true / false |
| chunk size / n action steps | 50 / 50 |
| AdamW | lr 2.5e-5，betas (0.9, 0.95)，weight decay 0.01，clip norm 1.0 |
| 官方 π0.5 scheduler | warmup 1000、decay 30000、末端 lr 2.5e-6 |
| 数据选择 / 验证 | 所有 episode；不划验证集，不运行验证 loss 或仿真评估。 |

学习率调度采用原官方 preset；其 decay 范围不等同于本次总更新步数。项目没有额外的自定义训练循环或梯度累积实现，`--gradient-accumulation-steps` 只接受 1。W&B 记录指标，关闭模型 artifact 上传，也不上传 checkpoint 到 Hugging Face。

## 7. 继续已有训练与迁移实验路径

继续本仓库内的同一实验：

```bash
./run.sh resume \
  --resume-from ./outputs/fr3_avocado_run2/checkpoints/last
```

`last` 表示最近一次已经保存的 checkpoint，不表示成功率最高的模型。也可固定数字目录，例如 `.../checkpoints/008000`。恢复到总步数 20,000 时，8,000 步断点还剩 12,000 次更新，**不是另外再跑 20,000 步**；不要传 `--steps 12000`。

已有旧工程的环境、数据和 checkpoint 都可以保留原位置。使用本仓库代码直接继续原实验的示例：

```bash
export PI05_ML_PYTHON="/root/autodl-tmp/mujoco_pick_place_pi05/.venv/bin/python"
./run.sh resume \
  --resume-from /root/autodl-tmp/Pi05-train/outputs/fr3_avocado_pi05/checkpoints/last
```

上面的命令沿用 checkpoint 保存的数据和输出绝对路径，后续 checkpoint 会继续写入原实验输出。这是操作者主动选择的续训，不是把旧实验拷进或提交到新仓库。

若数据或后续输出确实需要搬迁，显式指定路径，保持数据内容、batch size、总 steps 和学习率等实验设置不变：

```bash
./run.sh resume \
  --resume-from /原实验/checkpoints/008000 \
  --dataset /搬迁后的完整/fr3_avocado_v30 \
  --out /新的/实验续训输出目录
```

迁移只是资产/输出位置改变，不应换成新任务或新一批数据。默认保留保存的 W&B 设置及 run ID；不额外指定新的 project/run。恢复允许调整 workers、日志或保存间隔等运行设置，但本教程不改变它们。官方采样与恢复仍受原 batch/world size 等条件影响，不保证跨硬件运行逐位相同。

恢复必须保留整个数字 checkpoint：

```text
008000/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── train_config.json
│   ├── policy_preprocessor.json
│   ├── policy_postprocessor.json
│   └── processor 引用的统计状态文件
└── training_state/
    ├── optimizer_param_groups.json
    ├── optimizer_state.safetensors
    ├── scheduler_state.json
    ├── rng_state.safetensors
    └── training_step.json
```

仅部署时主要使用完整 `pretrained_model/`；续训还必须有同级 `training_state/`。旧自定义循环的 `COMPLETE.json` / `training_state/state.pt` 不等价于官方 checkpoint，入口会拒绝。Tokenizer 仍需保持其保存的缓存 ID 或绝对本地路径可用；resume 不接受 `--tokenizer` 来替换它。

## 8. 原生 checkpoint 加载修复与常见错误

首次初始化底座与恢复训练使用两种明确的严格加载路径：

- 新训练的 `pi05_base` 可以通过官方基础权重键名迁移适配到当前模型。
- `resume` 读取官方原生 checkpoint，按保存的原名严格加载，不再删除 `model.` 前缀，也不调用基础模型的 remapper。

这样保留训练后数值已经不同的 `lm_head.weight` 与 `embed_tokens.weight`。只有当前模型中确实为**同一个 Parameter/buffer 对象**的缺失别名可以补齐；独立参数缺失、额外 key、形状错误或共享参数冲突仍会失败，不会退回随机模型。

此前 `Conflicting remapped checkpoint key: ...embed_tokens.weight` 是把训练 checkpoint 再次按底座规则映射造成的加载错误。当前 resume 分支已修复；不要通过删 checkpoint key、强制绑定两份权重或关闭 strict 检查绕过。该修复不改变官方 optimizer、sampler、保存或恢复机制，也不会改写已有 checkpoint。

| 提示 | 处理 |
| --- | --- |
| 缺少基础模型目录中的 `config.json` | 使用完整的 `snapshots/<hash>`，不是 Hub cache 外层目录。 |
| 缺少 tokenizer / processor state | 保留训练时的完整 processor 文件和原 tokenizer 缓存/目录。 |
| 数据统计不是全量训练帧 | 使用已正确导出的无验证集数据，先通过 `check-train`；不要手改统计文件凑数。 |
| 恢复要求保持 steps / batch 等设置 | 去掉改变实验的参数；需要新超参数时创建新实验。 |
| CUDA OOM | 先核对同卡其他进程；程序不会替你结束它们。改变 batch 或梯度检查点会改变实验设置，不能无说明地混入原续训。 |
| Ctrl+C 中断后需要继续 | 从最近已保存的官方 checkpoint 恢复；尚未保存的更新不会由 `last` 自动补回。 |
