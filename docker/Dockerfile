# Versions match the working ROS underlay; build is explicit and starts no nodes.
FROM ros:humble-ros-base-jammy
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG LIBFRANKA_REF=155ddac80565975b58f53eb2feeb85b07a6b9e13
ARG FRANKA_ROS2_REF=dffd70ef838656b5ded3b8e460cc58d699d929f8
ARG FRANKA_DESCRIPTION_REF=4b8948e061c2ef8b2ea15d658cf35981c683f864
# Tags: libfranka 0.15.0, franka_ros2 v2.0.4, description 1.0.2.
# The v2.0.4 franka_ros2 source still declares package.xml version 2.0.2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates libeigen3-dev libfmt-dev libpoco-dev \
    python3-colcon-common-extensions python3-rosdep python3-vcstool python3-pip \
    python3-serial python3-numpy python3-yaml ros-humble-pinocchio \
    ros-humble-moveit-core ros-humble-moveit-msgs ros-humble-xacro \
    ros-humble-joint-state-publisher ros-humble-robot-state-publisher \
    ros-humble-ament-cmake-gmock ros-humble-ament-cmake-gtest \
    ros-humble-ament-lint-auto ros-humble-ament-lint-common \
    ros-humble-ament-cmake-clang-format ros-humble-ament-cmake-clang-tidy \
    ros-humble-ros2-control-test-assets && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir dynamixel-sdk==3.7.31 pyserial==3.5 tyro==0.9.35
COPY docker/prepare_sources.py docker/sources.lock.json docker/retry.sh /tmp/ros_sources/
COPY docker/vendor/ /tmp/ros_sources/vendor/
RUN python3 /tmp/ros_sources/prepare_sources.py --offline && \
    test "$LIBFRANKA_REF" = 155ddac80565975b58f53eb2feeb85b07a6b9e13 && \
    mkdir -p /tmp/libfranka && \
    tar -xzf /tmp/ros_sources/vendor/libfranka-0.15.0.tar.gz -C /tmp/libfranka --strip-components=1 && \
    rm -rf /tmp/libfranka/common && mkdir -p /tmp/libfranka/common && \
    tar -xzf /tmp/ros_sources/vendor/libfranka-common-cd38d0ec300b7e6864407d85d1e88e2fba31ccd5.tar.gz \
      -C /tmp/libfranka/common --strip-components=1 && cd /tmp/libfranka && \
    source /opt/ros/humble/setup.bash && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF && \
    cmake --build build --parallel 4 && cmake --install build && ldconfig && rm -rf /tmp/libfranka
RUN test "$FRANKA_ROS2_REF" = dffd70ef838656b5ded3b8e460cc58d699d929f8 && \
    test "$FRANKA_DESCRIPTION_REF" = 4b8948e061c2ef8b2ea15d658cf35981c683f864 && \
    mkdir -p /tmp/franka_ws/src/franka_ros2 /tmp/franka_ws/src/franka_description && \
    tar -xzf /tmp/ros_sources/vendor/franka_ros2-v2.0.4.tar.gz -C /tmp/franka_ws/src/franka_ros2 --strip-components=1 && \
    tar -xzf /tmp/ros_sources/vendor/franka_description-1.0.2.tar.gz -C /tmp/franka_ws/src/franka_description --strip-components=1 && \
    source /opt/ros/humble/setup.bash && \
    apt-get -o Acquire::Retries=5 update && \
    bash /tmp/ros_sources/retry.sh 3 apt-get install -y --no-install-recommends \
      ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-backward-ros \
      ros-humble-generate-parameter-library ros-humble-realtime-tools \
      ros-humble-rosidl-default-generators ros-humble-control-msgs ros-humble-std-srvs \
      ros-humble-urdf ros-humble-visualization-msgs ros-humble-ament-cmake-python && \
    cd /tmp/franka_ws && colcon build --install-base /opt/ros/humble/franka \
      --packages-up-to franka_gripper franka_robot_state_broadcaster franka_description \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    rm -rf /tmp/franka_ws /var/lib/apt/lists/*
COPY ros2_ws/src /tmp/custom_sources
RUN source /opt/ros/humble/setup.bash && source /opt/ros/humble/franka/setup.bash && \
    test -f /tmp/custom_sources/franka_fr3_arm_controllers/CMakeLists.txt && \
    test -f /tmp/custom_sources/pi05_gripper_driver/CMakeLists.txt && \
    rm -rf /tmp/custom_sources
RUN printf '%s\n' 'source /opt/ros/humble/setup.bash' \
    'source /opt/ros/humble/franka/setup.bash' >> /root/.bashrc
WORKDIR /workspace/ros2
CMD ["sleep", "infinity"]
