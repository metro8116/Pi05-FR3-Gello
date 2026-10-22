# 按阶段运行

[English](README.md) · 简体中文

命令完整示例在根 README 与 `docs/`，不提供“一键自动跑完整机器人流程”的脚本。阶段切换需要放置物体、检查机器人与相机、把数据上传到 GPU 主机，并由现场操作者决定何时启用模型。

统一入口 `./run.sh help`，各阶段参数可用 `./run.sh convert --help`、`./run.sh train --help`、`./run.sh serve --help`、`./run.sh deploy --help` 查看。`configs/*.env.example` 是两个机器角色的实际配置模板，复制成不带 `.example` 的文件后填写，真实路径与 SSH 配置不提交 Git。

## 仓库内置样例

| 样例 | 位置 | 内容与用途 |
| --- | --- | --- |
| 原始示范 | [data/raw/avocado_batch01/record_001/](../data/raw/avocado_batch01/record_001/) | 一条完整 schema-2 episode：304 帧对齐的状态/动作，15 FPS，包含双相机视频、保存的 RGB 图像、JSON/HDF5、时间戳与 metadata，用于查看格式和检查转换流程。 |
| 部署录像 | [deploy/recordings/20261022_155120_effebe48-177c-4bf9-b394-cbbf5fb0d370_e38d0c5b/](../deploy/recordings/20261022_155120_effebe48-177c-4bf9-b394-cbbf5fb0d370_e38d0c5b/) | 两路 640×480 RGB MP4，每路 554 帧、标称 30 FPS，以及逐帧元数据、录像元数据和编码日志，用于复盘真机执行。 |

这两个目录已明确纳入 Git 跟踪。完整数据集、模型权重、环境及后续新录像仍由默认忽略规则排除。自己采集时使用新的批次名，例如 `data/raw/avocado_batch02`，与内置原始样例分开保存。

部署录像不包含训练所需的采集 schema-2 状态/动作记录，不能直接加入示范数据集。单条原始样例用于验证流程；正式任务训练使用 [完整 LeRobot 数据集](https://huggingface.co/datasets/Metroooooooo/fr3-gello-avocado-v30) 或自己审查后的示范数据。

## 校验并转换原始样例

按 [环境教程](../docs/environments_zh.md) 配好 converter 环境后，从仓库根目录执行：

```bash
./run.sh validate \
  --input-dir ./data/raw/avocado_batch01 \
  --episodes record_001 \
  --output-dir ./data/lerobot/fr3_avocado_example_v30 \
  --report ./runs/example_raw_validation.json

./run.sh convert \
  --input-dir ./data/raw/avocado_batch01 \
  --episodes record_001 \
  --repo-id local/fr3_avocado_example \
  --output-dir ./data/lerobot/fr3_avocado_example_v30
```

校验应选中 **1 条 episode / 304 帧**，跳过和无效记录均为 0；该步骤不会生成转换后的数据集。转换会新建 `fr3_avocado_example_v30`，输出目录已存在时拒绝覆盖，与完整训练数据集分开保存。`--input-dir` 指向直接包含 `record_001` 的批次目录，而不是外层 `data/raw`。

在已配置 ML 环境的机器上，可检查训练侧读取，不加载模型权重，也不开始训练：

```bash
./run.sh check-train \
  --dataset ./data/lerobot/fr3_avocado_example_v30 \
  --data-only
```

完整数据约定、人工审查裁剪和转换来源记录见 [数据教程](../docs/data_zh.md)。
