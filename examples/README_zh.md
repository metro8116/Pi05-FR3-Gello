# 按阶段运行

[English](README.md) · 简体中文

命令完整示例在根 README 与 `docs/`，不提供“一键自动跑完整机器人流程”的脚本。阶段切换需要放置物体、检查机器人与相机、把数据上传到 GPU 主机，并由现场操作者决定何时启用模型。

统一入口 `./run.sh help`，各阶段参数可用 `./run.sh convert --help`、`./run.sh train --help`、`./run.sh serve --help`、`./run.sh deploy --help` 查看。`configs/*.env.example` 是两个机器角色的实际配置模板，复制成不带 `.example` 的文件后填写，真实路径与 SSH 配置不提交 Git。
