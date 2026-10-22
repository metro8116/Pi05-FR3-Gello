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
| `src/fr3_pi05`、训练入口 | 用户当前 Pi05-train 工程 | 本地功能迁入与集成改动；发布前由权利人明确授权条款 |
| `deploy/` | 用户当前 Pi05-deploy 工程 | 本地功能迁入与集成改动；派生驱动仍按组件许可 |
| 转换器与集成脚本/教程 | 用户本次工作流与新集成 | 未将第三方未授权部分重新许可；发布前统一确认原创部分授权 |

GELLO 上游：[wuphilipp/gello_software](https://github.com/wuphilipp/gello_software)。仓库根 MIT 文本留存在 `licenses/gello_software-MIT.txt` 作来源记录，不将其扩展到其他来源。

运行依赖不随源码打包：LeRobot、Transformers、PyTorch、libfranka、franka_ros2、RealSense SDK、FFmpeg/imageio-ffmpeg 各遵循自身许可。libfranka 的源码构建、镜像分发也需遵守其对应版本许可与告知要求。模型 `lerobot/pi05_base`、PaliGemma tokenizer、数据与实际视频分别适用其自身使用/分发条件，不能用代码仓库许可代替。

本次只在本地新建代码仓库，没有上传 GitHub。公开发布前应向采集上游权利人确认再分发条款，或用得到授权/重新实现的采集部分替换；同时明确项目原创文件的许可。此说明记录已有事实，不代表已经获得第三方授权。
