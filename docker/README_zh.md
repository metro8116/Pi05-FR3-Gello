# ROS 镜像源码准备与断线重试

[English](README.md) · 简体中文

`./run.sh ros-image` 会先在主机运行 `prepare_sources.py`，准备四个固定提交的官方源码压缩包。`sources.lock.json` 记录版本、提交、来源 URL 与 SHA256，包括 libfranka 的精确 common 子模块提交；不克隆完整 Git 历史，不关闭 TLS 验证。

缓存位于本仓库 `docker/vendor/`，四份官方包共约 36.7 MB。缓存经过校验后复用，加入 Docker build context，但被 Git 忽略；发布仓库时不上传缓存。首次下载失败会重试，临时文件不会当成完整源码。已有缓存校验失败时保留并报错，不静默覆盖它。

```bash
./run.sh ros-image
```

Docker 中只使用已经校验的本地源码，并再次执行离线校验；libfranka/common、franka_ros2 和 franka_description 的版本保持固定。系统 apt 依赖仍需要联网；后续 Franka ROS 依赖阶段设有重试，首个基础 apt 阶段失败时可重新运行统一入口。镜像明确安装真机控制所需依赖，不再通过 rosdep 请求 GitHub 索引；只构建 franka_gripper、franka_robot_state_broadcaster、franka_description 及其源码依赖闭包，不包含未使用的 Gazebo/MoveIt 示例。项目自定义 controller 所需 MoveIt Core/Msgs 仍保留。首次编译仍需等待，源码下载通过并不代表全部依赖安装已完成。

手动准备或复用另一目录中相同版本的源码包：

```bash
python3 docker/prepare_sources.py
python3 docker/prepare_sources.py --offline
python3 docker/prepare_sources.py --cache-dir /实际路径/源码缓存
```

`--cache-dir` 可重复指定，不会扫描旧工程或修改导入目录。锁文件也接受经过审计的三份历史完整 Git 源码包：它们的 tracked files 无修改、HEAD 与固定提交一致；common 仍由单独锁定的第四包提供。其他来源或哈希不匹配的包不会使用。

网络中断后重新执行 `./run.sh ros-image` 即可，保留 Docker 已完成层；无需重建 Python 环境，也无需执行 Docker prune。直接调用 compose build 时必须先运行源码准备工具；统一入口会自动处理。
