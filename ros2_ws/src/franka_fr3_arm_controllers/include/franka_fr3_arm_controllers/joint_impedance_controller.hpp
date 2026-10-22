// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <mutex>
#include <atomic>
#include "franka_fr3_arm_controllers/motion_generator.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_fr3_arm_controllers {

/**
 * Controller to move the robot to a desired joint position.
 */
class JointImpedanceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  std::string arm_id_;
  std::string namespace_prefix_;
  std::string robot_description_;
  const int num_joints = 7;
  Vector7d q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  double k_alpha_;
  bool relative_start_{true};
  bool move_to_start_position_finished_{false};
  bool motion_generator_initialized_{false};
  rclcpp::Time start_time_;
  std::unique_ptr<MotionGenerator> motion_generator_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_ = nullptr;
  bool gello_position_values_valid_ = false;
  bool input_fault_ = false;
  std::atomic<bool> shutdown_requested_{false};
  rclcpp::TimerBase::SharedPtr shutdown_timer_;
  struct GelloInputSample {
    std::array<double, 7> positions{0, 0, 0, 0, 0, 0, 0};
    rclcpp::Time receipt_time{0, 0, RCL_ROS_TIME};
    bool valid{false};
    bool fault{false};
  };
  GelloInputSample input_buffer_;
  std::mutex gello_state_mutex_;
  std::array<double, 7> gello_position_values_{0, 0, 0, 0, 0, 0, 0};
  std::array<double, 7> gello_start_position_values_{0, 0, 0, 0, 0, 0, 0};
  std::array<double, 7> robot_start_position_values_{0, 0, 0, 0, 0, 0, 0};
  rclcpp::Time last_joint_state_time_;

  // Fixed-size handoff only; JSON serialization and ROS publishing run on the timer.
  struct ArmCommandSnapshot {
    uint64_t sequence{0};
    int64_t activation_id_ns{0};
    int64_t stamp_ns{0};
    std::array<double, 7> q_measured{};
    std::array<double, 7> dq_measured{};
    std::array<double, 7> q_target{};
    std::array<double, 7> tau_command{};
    std::array<double, 7> gello_q{};
    std::array<double, 7> robot_start_q{};
    std::array<double, 7> gello_start_q{};
    bool relative_start{true};
    bool valid{false};
  };
  ArmCommandSnapshot arm_command_snapshot_;
  std::mutex arm_command_mutex_;
  std::atomic<bool> arm_command_active_{false};
  uint64_t update_sequence_{0};
  std::atomic<int64_t> activation_id_ns_{0};
  uint64_t last_published_sequence_{0};
  int64_t last_published_activation_id_ns_{0};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr arm_command_publisher_;
  rclcpp::TimerBase::SharedPtr arm_command_timer_;

  void publishArmCommand_();
  Vector7d calculateTauDGains_(const Vector7d& q_goal);
  bool validateGains_(const std::vector<double>& gains, const std::string& gains_name);
  bool initializeMotionGenerator_();
  void updateJointStates_();
  void validateGelloPositions_(const sensor_msgs::msg::JointState& msg);
  void jointStateCallback_(const sensor_msgs::msg::JointState msg);
};

}  // namespace franka_fr3_arm_controllers
