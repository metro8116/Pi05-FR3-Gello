# 数据接口、备份与 LeRobot v3.0 转换

[English](data.md) · 简体中文

下文示例使用 `data/raw/avocado_batch01/record_NNN` → `data/lerobot/fr3_avocado_v30`。Git 已包含 `data/raw/avocado_batch01/record_001/` 一条完整原始样例；完整采集批次与转换后的数据集单独准备，由默认忽略规则排除。单条样例转换到独立输出目录的步骤见 [内置样例](../examples/README_zh.md)，正式训练可下载 [完整 LeRobot v3.0 数据集](https://huggingface.co/datasets/Metroooooooo/fr3-gello-avocado-v30)。相机录像、JSON、HDF5、timestamp 与 metadata 是一整条记录，不能只拿出一部分转换。

## 固定接口

| 项目 | 约定 |
| --- | --- |
| task | `Put the avocado in the basket.` |
| `observation.state` | FR3 实测 q1…q7（rad）+ 实测夹爪总宽（m） |
| `action` | FR3 controller 的绝对 q1…q7 目标（rad）+ 夹爪最近接受的目标总宽（m） |
| state/action | float32，各 8 维；不是 GELLO 原始读数，不是 0–1 夹爪闭合率 |
| camera_1 | `observation.images.camera_1`，serial `233522074047` |
| camera_2 | `observation.images.camera_2`，serial `405622074065` |
| 图像 | RGB 640×480，转换后加载为 CHW 浮点 [0,1] |
| FPS | RGB 采集 30，记录/训练 15，无隐式重采样 |
| action 模式 | absolute，`use_relative_actions=false` |
| 当前夹爪执行 | Move 0.04 m/s，总宽 0–0.08 m，模型不预测力/速度 |

遥操控制器以当前 FR3/GELLO 姿态作为相对零点，但记录的是**由此形成的绝对 FR3 指令**；部署不能再次给预测结果加起始偏置。实测夹爪总宽与目标总宽在夹物时不同是合理的，不能把目标改成实测宽度来掩盖差异。

## 先备份，再审查

关闭录制与遥操程序，确认文件不再写入，然后对原始目录做一份独立拷贝到足够空间的目录。不要用同一目标覆盖旧备份；保留原始 metadata 与日志用于追溯。可使用 `cp -a` 或 `rsync -a`，先核对源/目标，备份放在 `data/` 里也不会上传 Git。

逐条看两路 MP4，检查抓取是否保持、是否确实放下、是否完整收手；同时检查动作/时间戳与相机序号。现场失败的 episode 标为 failure/aborted，即使文件本身完整也不应被当成功示范。不要因夹爪 Move 的 `Command failed!` 一条消息就判断整条数据可用或不可用：转换器保留原始执行诊断，任务成功还需结合实际视频和测量。

首尾近静止帧可在**人工审查后**裁剪，保持完整接近、抓住、放置、开爪和收手。中间等待可能属于接触/释放行为，不用单纯“动作小”自动判为无用帧。本仓库不自动删除或修改原始文件。

## 只读校验

```bash
./run.sh validate --input-dir ./data/raw/avocado_batch01 \
  --report ./runs/raw_validation.json
```

`--input-dir` 必须直接包含 `record_*` 目录；使用自己的数据时替换批次路径。`--dry-run` 在入口中自动添加：校验 schema、JSON/HDF5 一致性、8 维数据、完整性/任务/标签、源时间新鲜度、同一命令身份的内容一致性、相机顺序、视频帧数、曝光/接收时间、FPS 与帧间隙。依赖缺失会直接失败，不能将 “Selected 0” 当成通过。

默认只选 `status=complete`、`outcome=success`、满足接口的记录。错误段不会被静默修复后混入。输出应有非空 selected 和正数帧数；查看 report 里的 skipped/invalid 原因，再决定是否排除或重采。

## 导出

```bash
./run.sh convert --input-dir ./data/raw/avocado_batch01 \
  --repo-id local/fr3_avocado_pi05 \
  --output-dir ./data/lerobot/fr3_avocado_v30
```

输出已存在会拒绝覆盖。先写临时 staging、成功后发布新目录，原始文件不变。默认 `--val-ratio 0`，本任务全部有效 episode 用于训练。支持 `--episodes record_001 record_002` 显式选择；修改筛选条件会形成新的数据版本，训练开始后不要就地替换它。

输出包含 Parquet、两路视频、tasks/episodes/stats 元数据、转换 manifest、源文件摘要及每帧来源时间戳。训练接口读取这些来源约定并独立检查全量标量统计。保留**整个输出目录**，不要移动或单独重编码其中视频。

转换与训练均固定 LeRobot 0.6.1，磁盘格式为 v3.0。转换使用 CPU PyTorch，AutoDL 训练使用 CUDA PyTorch，因此仍使用独立环境，详见 [requirements/README.md](../requirements/README_zh.md)。

## 可选人工裁剪计划

转换器支持 `--trim-plan /实际路径/plan.json`。计划 schema 为 `fr3_tail_trim_v1`、fps=15，每个被选择 record 恰好一项，指定 `original_frames`、`keep_start_inclusive`、`keep_end_exclusive`、非空 `reason`、`review`，以及完整 `source_files` 指纹字典（每文件 size/mtime_ns/sha256）。区间为左闭右开。

source_files 必须来自审查时的原始完整文件，不能用裁剪后的摘要代替。工具先验证整条原始数据，再切片 state、action、图像与来源时间戳，重新计算统计，并记录原始帧索引与裁剪计划。计划不覆盖全部选中记录、指纹不匹配或保留区间太短都会拒绝转换。未裁剪的条目也需明确保留 [0,original_frames)。

本次没有为现有真实数据新建裁剪计划，也没有再次清洗。已有清洗好的 v3.0 数据直接复制/指定给训练即可，目录无需加 clean 后缀。如何单独复制并上传见 [训练教程](training_zh.md)。

## 转换后的检查

在 ML 环境运行：

```bash
./run.sh check-train --dataset ./data/lerobot/fr3_avocado_v30 --data-only
```

它检查训练读取、两相机样本、50 步 action chunk 和 episode 末尾 padding，核对统计没有跨 episode。只有校验通过才进入训练。原始数据可以留在机器人电脑，不需要上传 AutoDL；完整 v3.0 和模型/tokenizer 才是训练输入。

部署时自动保存的相机视频用于复盘，单独存放在部署录像目录；不具备采集 schema-2 的 state/action/因果对齐，不能直接混入示范数据集。
