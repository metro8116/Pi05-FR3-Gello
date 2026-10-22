// Copyright (c) 2026
// Licensed under the Apache License, Version 2.0; see LICENSE.
#include <memory>
#include <pi05_gripper_driver/gripper_action_server.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<pi05_gripper_driver::GripperActionServer>();
  // Two callback groups need two executor threads; changing only the executor
  // leaves callbacks in the same default mutually-exclusive group serialized.
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 2);
  executor.add_node(node);
  executor.spin();
  executor.remove_node(node);
  rclcpp::shutdown();
  return 0;
}
