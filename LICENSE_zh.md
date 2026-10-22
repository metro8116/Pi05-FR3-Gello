# 组件来源与许可

[English](LICENSE.md) · 简体中文

本仓库是多个来源的集成工程，**不声明整个目录统一为 MIT 或 Apache-2.0**。各文件原有版权头、LICENSE 与 NOTICE 保留并优先适用。

| 组件 | 来源 | 许可/状态 |
| --- | --- | --- |
| `ros2_ws/src/franka_gello_state_publisher` | 本地 GELLO ROS 工程 | 保留组件 MIT LICENSE |
| `ros2_ws/src/franka_fr3_arm_controllers` | Franka ROS 示例及本地控制改动 | 保留 Apache-2.0 LICENSE/NOTICE |
| `ros2_ws/src/franka_gripper_manager` | 本地 Franka ROS 工程 | 保留 Apache-2.0 LICENSE/NOTICE |
| `ros2_ws/src/pi05_gripper_driver` | Franka gripper 派生的独立驱动 | 保留 Apache-2.0 LICENSE/NOTICE |
| `capture/` | [renshaojie233/data-collection](https://github.com/renshaojie233/data-collection) 及当前实验改动 | 迁移时上游根目录未见明确 LICENSE；详见 `capture/SOURCE.txt` |
| `src/fr3_pi05`、训练入口 | 用户当前 Pi05-train 工程 | 本地功能迁入与集成改动；原创部分尚未指定许可条款 |
| `deploy/` | 用户当前 Pi05-deploy 工程 | 本地功能迁入与集成改动；派生驱动仍按组件许可 |
| 转换器与集成脚本/教程 | 用户本次工作流与新集成 | 未将第三方未授权部分重新许可；原创部分尚未指定许可条款 |

GELLO 上游：[wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)。仓库根 MIT 文本留存在 `licenses/gello_software-MIT.txt` 作来源记录，不将其扩展到其他来源。

运行依赖不随源码打包：LeRobot、Transformers、PyTorch、libfranka、franka_ros2、RealSense SDK、FFmpeg/imageio-ffmpeg 各遵循自身许可。libfranka 的源码构建、镜像分发也需遵守其对应版本许可与告知要求。模型 `lerobot/pi05_base`、PaliGemma tokenizer、数据与实际视频分别适用其自身使用/分发条件，不能用代码仓库许可代替。

本仓库已发布于 GitHub。发布行为本身不构成许可授予，也不代表已获得第三方代码的再分发授权。采集上游的再分发条款仍未确认，需要权利人明确，或用已获授权、独立实现的替代方案替换该组件。项目原创文件也仍未指定许可条款。此说明记录来源与待解决的许可问题，不代表已经获得第三方授权。
