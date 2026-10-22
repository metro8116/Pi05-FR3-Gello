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

#include <franka_fr3_arm_controllers/joint_impedance_controller.hpp>

#include <Eigen/Eigen>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

using std::placeholders::_1;

namespace franka_fr3_arm_controllers {

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/position");
    config.names.push_back(namespace_prefix_ + arm_id_ + "_joint" + std::to_string(i) +
                           "/velocity");
  }
  return config;
}

controller_interface::return_type JointImpedanceController::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& /*period*/) {
  ++update_sequence_;
  // Cleanup may release interfaces after the first fault. The zero efforts are
  // written before publishing this latch, so subsequent cycles must not access them.
  if (shutdown_requested_.load()) {
    return controller_interface::return_type::ERROR;
  }
  // Never wait on the ROS callback from the real-time update thread. If a callback is
  // writing, retain the previous complete sample; the watchdog still checks its age.
  {
    std::unique_lock<std::mutex> lock(gello_state_mutex_, std::try_to_lock);
    if (lock.owns_lock()) {
      gello_position_values_ = input_buffer_.positions;
      gello_position_values_valid_ = input_buffer_.valid;
      input_fault_ = input_fault_ || input_buffer_.fault;
      last_joint_state_time_ = input_buffer_.receipt_time;
    }
  }
  const double input_age = (get_node()->now() - last_joint_state_time_).seconds();
  if (input_fault_ || input_age < 0.0 || input_age > 0.5) {
    input_fault_ = true;
    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(0.0);
    }
    if (!shutdown_requested_.exchange(true)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Stopping controller: invalid or stale GELLO input (age: %f s)", input_age);
    }
    // The non-real-time timer performs shutdown. Never run controller-manager
    // cleanup here: a pending switch service may hold its lock while waiting for update().
    return controller_interface::return_type::ERROR;
  }
  updateJointStates_();
  Vector7d q_goal;
  Vector7d tau_d_calculated;

  if (!motion_generator_initialized_) {
    // After starting the controller we wait for valid joint states from the input topic
    // Until we get valid joint states we will send zero torques to the robot
    // to allow the user to reposition the robot
    motion_generator_initialized_ = initializeMotionGenerator_();

    if (!motion_generator_initialized_) {
      for (int i = 0; i < num_joints; ++i) {
        command_interfaces_[i].set_value(0.0);
      }

      return controller_interface::return_type::OK;
    }
  }

  if (!move_to_start_position_finished_) {
    // We have received valid joint states and initialized the motion generator
    // Now we move smoothly to the first joint position received from the input topic
    auto trajectory_time = this->get_node()->now() - start_time_;
    auto motion_generator_output = motion_generator_->getDesiredJointPositions(trajectory_time);
    move_to_start_position_finished_ = motion_generator_output.second;

    q_goal = motion_generator_output.first;
  }

  if (move_to_start_position_finished_) {
    // After reaching the start position we follow the joint position from the input topic
    // This is the normal operation mode of the controller
    for (int i = 0; i < num_joints; ++i) {
      if (relative_start_) {
        q_goal(i) = robot_start_position_values_[i] +
                    (gello_position_values_[i] - gello_start_position_values_[i]);
      } else {
        q_goal(i) = gello_position_values_[i];
      }
    }
  }

  tau_d_calculated = calculateTauDGains_(q_goal);

  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }

  // Observation, target and torque all belong to this exact control update.
  // If the non-RT timer owns the handoff lock, skip telemetry without waiting.
  {
    std::unique_lock<std::mutex> lock(arm_command_mutex_, std::try_to_lock);
    if (lock.owns_lock()) {
      arm_command_snapshot_.sequence = update_sequence_;
      arm_command_snapshot_.activation_id_ns = activation_id_ns_.load();
      arm_command_snapshot_.stamp_ns =
          time.nanoseconds() > 0 ? time.nanoseconds() : get_node()->now().nanoseconds();
      for (int i = 0; i < num_joints; ++i) {
        arm_command_snapshot_.q_measured[i] = q_(i);
        arm_command_snapshot_.dq_measured[i] = dq_(i);
        arm_command_snapshot_.q_target[i] = q_goal(i);
        arm_command_snapshot_.tau_command[i] = tau_d_calculated(i);
      }
      arm_command_snapshot_.gello_q = gello_position_values_;
      arm_command_snapshot_.robot_start_q = robot_start_position_values_;
      arm_command_snapshot_.gello_start_q = gello_start_position_values_;
      arm_command_snapshot_.relative_start = relative_start_;
      arm_command_snapshot_.valid = true;
    }
  }

  return controller_interface::return_type::OK;
}

void JointImpedanceController::jointStateCallback_(const sensor_msgs::msg::JointState msg) {
  std::lock_guard<std::mutex> lock(gello_state_mutex_);
  if (input_buffer_.receipt_time.seconds() == 0.0) {
    return;
  }

  if (msg.position.size() < gello_position_values_.size()) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Received joint state size is smaller than expected size.");
    input_buffer_.fault = true;
    return;
  }

  if (!std::all_of(msg.position.begin(), msg.position.begin() + num_joints,
                   [](double value) { return std::isfinite(value); })) {
    input_buffer_.fault = true;
    RCLCPP_ERROR(get_node()->get_logger(), "Received non-finite GELLO joint positions");
    return;
  }
  validateGelloPositions_(msg);
  if (!input_buffer_.valid) {
    input_buffer_.fault = true;
    return;
  }
  std::copy(msg.position.begin(), msg.position.begin() + gello_position_values_.size(),
            input_buffer_.positions.begin());

  input_buffer_.receipt_time = get_node()->now();
}

CallbackReturn JointImpedanceController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<bool>("relative_start", true);
    auto_declare<std::string>("command_topic", "gello/joint_states");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  namespace_prefix_ = get_node()->get_namespace();
  if (namespace_prefix_ == "/" || namespace_prefix_.empty()) {
    namespace_prefix_.clear();
  } else {
    // Remove leading slash and add trailing underscore
    namespace_prefix_ = namespace_prefix_.substr(1) + "_";
  }

  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  auto k_alpha = get_node()->get_parameter("k_alpha").as_double();
  relative_start_ = get_node()->get_parameter("relative_start").as_bool();

  if (!validateGains_(k_gains, "k_gains") || !validateGains_(d_gains, "d_gains")) {
    return CallbackReturn::FAILURE;
  }

  for (int i = 0; i < num_joints; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
  }

  if (k_alpha < 0.0 || k_alpha > 1.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_alpha should be in the range [0, 1]");
    return CallbackReturn::FAILURE;
  }

  k_alpha_ = k_alpha;

  dq_filtered_.setZero();

  auto parameters_client =
      std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  const auto command_topic = get_node()->get_parameter("command_topic").as_string();
  if (command_topic.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "command_topic must not be empty");
    return CallbackReturn::FAILURE;
  }
  joint_state_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      command_topic, 1,
      [this](const sensor_msgs::msg::JointState& msg) { jointStateCallback_(msg); });

  // Humble controller_manager 2.53 only logs an update ERROR. Explicit context
  // shutdown invokes its pre-shutdown controller/hardware cleanup and stops FCI.
  shutdown_timer_ = get_node()->create_wall_timer(std::chrono::milliseconds(10), [this]() {
    if (shutdown_requested_.load()) {
      rclcpp::shutdown(get_node()->get_node_base_interface()->get_context(),
                        "Invalid or stale GELLO input");
    }
  });

  // Use a regular publisher and explicit capture lifecycle gating, including in
  // controller tests which invoke on_activate directly rather than a node transition.
  arm_command_publisher_ = rclcpp::create_publisher<std_msgs::msg::String>(
      *get_node(), "/gello_capture/arm_command", rclcpp::QoS(10));
  arm_command_timer_ = get_node()->create_wall_timer(
      std::chrono::milliseconds(20), [this]() { publishArmCommand_(); });

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (shutdown_requested_.load()) {
    return CallbackReturn::FAILURE;
  }
  arm_command_active_.store(false);
  std::lock_guard<std::mutex> lock(gello_state_mutex_);
  motion_generator_initialized_ = false;
  move_to_start_position_finished_ = false;
  gello_position_values_valid_ = false;
  input_fault_ = false;
  motion_generator_.reset();
  gello_start_position_values_.fill(0.0);
  robot_start_position_values_.fill(0.0);
  last_joint_state_time_ = get_node()->now();
  input_buffer_ = GelloInputSample{};
  input_buffer_.receipt_time = last_joint_state_time_;
  dq_filtered_.setZero();
  start_time_ = this->get_node()->now();
  update_sequence_ = 0;
  activation_id_ns_.store(start_time_.nanoseconds());
  {
    std::unique_lock<std::mutex> command_lock(arm_command_mutex_, std::try_to_lock);
    if (command_lock.owns_lock()) {
      arm_command_snapshot_ = ArmCommandSnapshot{};
      last_published_sequence_ = 0;
      last_published_activation_id_ns_ = 0;
    }
  }
  // If the timer currently owns the snapshot, its old generation is already invalid.
  arm_command_active_.store(true);

  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_command_active_.store(false);
  std::unique_lock<std::mutex> lock(arm_command_mutex_, std::try_to_lock);
  if (lock.owns_lock()) arm_command_snapshot_.valid = false;
  return CallbackReturn::SUCCESS;
}

void JointImpedanceController::publishArmCommand_() {
  if (!arm_command_active_.load() || shutdown_requested_.load()) {
    return;
  }
  ArmCommandSnapshot snapshot;
  {
    std::lock_guard<std::mutex> lock(arm_command_mutex_);
    if (!arm_command_snapshot_.valid ||
        arm_command_snapshot_.activation_id_ns != activation_id_ns_.load() ||
        (arm_command_snapshot_.activation_id_ns == last_published_activation_id_ns_ &&
         arm_command_snapshot_.sequence == last_published_sequence_)) {
      return;
    }
    snapshot = arm_command_snapshot_;
  }

  // All allocations, JSON formatting and publishing stay off the RT update loop.
  std::ostringstream json;
  json.imbue(std::locale::classic());
  json << std::setprecision(std::numeric_limits<double>::max_digits10);
  json << "{\"schema_version\":1,\"sequence\":" << snapshot.sequence
       << ",\"activation_id\":\"" << snapshot.activation_id_ns << "\",\"stamp_ns\":"
       << snapshot.stamp_ns
       << ",\"joint_names\":[\"fr3_joint1\",\"fr3_joint2\",\"fr3_joint3\",\"fr3_joint4\","
          "\"fr3_joint5\",\"fr3_joint6\",\"fr3_joint7\"]";
  const auto append_array = [&json](const char* name, const std::array<double, 7>& values) {
    json << ",\"" << name << "\":[";
    for (size_t i = 0; i < values.size(); ++i) {
      if (i != 0) json << ',';
      json << values[i];
    }
    json << ']';
  };
  append_array("q_measured", snapshot.q_measured);
  append_array("dq_measured", snapshot.dq_measured);
  append_array("q_target", snapshot.q_target);
  append_array("tau_command", snapshot.tau_command);
  append_array("gello_q", snapshot.gello_q);
  append_array("robot_start_q", snapshot.robot_start_q);
  append_array("gello_start_q", snapshot.gello_start_q);
  json << ",\"relative_start\":" << (snapshot.relative_start ? "true" : "false") << '}';
  std_msgs::msg::String message;
  message.data = json.str();

  // Serialize outside the handoff lock, then recheck activation before emitting.
  // The RT side always uses try_lock, so this publisher cannot block control.
  std::lock_guard<std::mutex> lock(arm_command_mutex_);
  if (!arm_command_active_.load() || shutdown_requested_.load() ||
      !arm_command_snapshot_.valid ||
      activation_id_ns_.load() != snapshot.activation_id_ns ||
      arm_command_snapshot_.activation_id_ns != snapshot.activation_id_ns) {
    return;
  }
  arm_command_publisher_->publish(message);
  last_published_sequence_ = snapshot.sequence;
  last_published_activation_id_ns_ = snapshot.activation_id_ns;
}

auto JointImpedanceController::calculateTauDGains_(const Vector7d& q_goal) -> Vector7d {
  dq_filtered_ = (1 - k_alpha_) * dq_filtered_ + k_alpha_ * dq_;
  Vector7d tau_d_calculated;
  tau_d_calculated = k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);

  return tau_d_calculated;
}

bool JointImpedanceController::validateGains_(const std::vector<double>& gains,
                                              const std::string& gains_name) {
  if (gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s parameter not set", gains_name.c_str());
    return false;
  }

  if (gains.size() != static_cast<uint>(num_joints)) {
    RCLCPP_FATAL(get_node()->get_logger(), "%s should be of size %d but is of size %ld",
                 gains_name.c_str(), num_joints, gains.size());
    return false;
  }

  return true;
}

void JointImpedanceController::validateGelloPositions_(const sensor_msgs::msg::JointState& msg) {
  const double max_time_diff = 0.5;
  auto current_time = get_node()->now();
  auto time_since_last_joint_state = (current_time - input_buffer_.receipt_time).seconds();
  auto time_since_msg_stamp = (current_time - msg.header.stamp).seconds();
  input_buffer_.valid =
      (time_since_last_joint_state >= 0.0 && time_since_last_joint_state < max_time_diff &&
       time_since_msg_stamp >= 0.0 && time_since_msg_stamp < max_time_diff);
  if (!input_buffer_.valid) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Gello position values are not valid. Time since last joint state: %f // Time "
                "since message stamp: %f",
                time_since_last_joint_state, time_since_msg_stamp);
  }
}

void JointImpedanceController::updateJointStates_() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

bool JointImpedanceController::initializeMotionGenerator_() {
  if (!gello_position_values_valid_) {
    // Only send a warning once every 10 seconds in order not to spam the log
    RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 10 * 1000,
                         "Waiting for valid joint states...");
    return false;
  }

  Vector7d q_goal;
  updateJointStates_();
  if (relative_start_) {
    for (int i = 0; i < num_joints; ++i) {
      robot_start_position_values_[i] = q_(i);
      gello_start_position_values_[i] = gello_position_values_[i];
    }
    move_to_start_position_finished_ = true;
    RCLCPP_INFO(get_node()->get_logger(),
                "Relative start enabled. Holding current robot q as zero: [%f, %f, %f, %f, %f, %f, %f]",
                q_(0), q_(1), q_(2), q_(3), q_(4), q_(5), q_(6));
    RCLCPP_INFO(get_node()->get_logger(),
                "Captured GELLO zero: [%f, %f, %f, %f, %f, %f, %f]",
                gello_start_position_values_[0], gello_start_position_values_[1],
                gello_start_position_values_[2], gello_start_position_values_[3],
                gello_start_position_values_[4], gello_start_position_values_[5],
                gello_start_position_values_[6]);
    return true;
  }

  for (int i = 0; i < num_joints; ++i) {
    q_goal(i) = gello_position_values_[i];
  }
  RCLCPP_INFO(get_node()->get_logger(), "q_goal of motion generator: [%f, %f, %f, %f, %f, %f, %f]",
              q_goal(0), q_goal(1), q_goal(2), q_goal(3), q_goal(4), q_goal(5), q_goal(6));

  const double motion_generator_speed_factor = 0.2;
  motion_generator_ = std::make_unique<MotionGenerator>(motion_generator_speed_factor, q_, q_goal);
  return true;
}

}  // namespace franka_fr3_arm_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_fr3_arm_controllers::JointImpedanceController,
                       controller_interface::ControllerInterface)
