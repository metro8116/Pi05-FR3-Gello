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

#include "include/test_joint_impedance_controller.hpp"
#include <limits>

TEST_F(JointImpedanceControllerTest, TestInitialization) {
  EXPECT_EQ(controller_->on_init(), CallbackReturn::SUCCESS);
}

TEST_F(JointImpedanceControllerTest, TestValidConfiguration) {
  controller_->on_init();
  rclcpp_lifecycle::State state;

  EXPECT_EQ(controller_->on_configure(state), CallbackReturn::SUCCESS);
}

TEST_F(JointImpedanceControllerTest, TestActivate) {
  EXPECT_EQ(startController(), CallbackReturn::SUCCESS);
}

TEST_F(JointImpedanceControllerTest, TestUpdateMotionGeneratorOnly) {
  static const std::vector<double> kInitialRobotPosition = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  static constexpr double kExpectedValue = -0.05;
  static constexpr double kTolerance = 0.01;

  setRobotPosition(kInitialRobotPosition);
  startController();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  rclcpp::Time time;
  rclcpp::Duration period = rclcpp::Duration::from_seconds(1.0);
  EXPECT_EQ(controller_->update(time, period), controller_interface::return_type::OK);

  for (size_t i = 0; i < joint_commands_.size(); ++i) {
    EXPECT_NEAR(joint_commands_[i], kExpectedValue, kTolerance);
  }
}

TEST_F(JointImpedanceControllerTest, TestUpdateMotionGeneratorAndGelloPositionValues) {
  static const std::vector<double> kInitialRobotPosition = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  static constexpr double kExpectedValue = -0.05;
  static constexpr double kTolerance = 0.01;

  setRobotPosition(kInitialRobotPosition);
  startController();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  rclcpp::Time time;
  rclcpp::Duration period = rclcpp::Duration::from_seconds(1.0);
  EXPECT_EQ(controller_->update(time, period), controller_interface::return_type::OK);

  for (size_t i = 0; i < joint_commands_.size(); ++i) {
    EXPECT_NEAR(joint_commands_[i], kExpectedValue, kTolerance);
  }
}

TEST_F(JointImpedanceControllerTest, TestUpdateGelloPositionValuesOnly) {
  static const std::vector<double> kInitialRobotPosition = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  static constexpr double kExpectedValue = -0.075;
  static constexpr double kTolerance = 0.01;

  setRobotPosition(kInitialRobotPosition);
  startController();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  rclcpp::Time time;
  rclcpp::Duration period = rclcpp::Duration::from_seconds(1.0);
  EXPECT_EQ(controller_->update(time, period), controller_interface::return_type::OK);
  period = rclcpp::Duration::from_seconds(1.1);
  EXPECT_EQ(controller_->update(time, period), controller_interface::return_type::OK);

  for (size_t i = 0; i < joint_commands_.size(); ++i) {
    EXPECT_NEAR(joint_commands_[i], kExpectedValue, kTolerance);
  }
}

TEST_F(JointImpedanceControllerTest, TestUpdateInvalidGelloPositionValues) {
  startController();
  mock_joint_state_publisher_->stopPublishing();
  mock_joint_state_publisher_->publishPositions(std::vector<double>(7, 1.0), 1.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(controller_->update(controller_->get_node()->now(),
                               rclcpp::Duration::from_seconds(0.001)),
            controller_interface::return_type::ERROR);
  for (double command : joint_commands_) EXPECT_DOUBLE_EQ(command, 0.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(rclcpp::ok());
}

TEST_F(JointImpedanceControllerTest, TestNonFiniteInputStopsWithoutWritingNan) {
  startController();
  mock_joint_state_publisher_->stopPublishing();
  auto positions = std::vector<double>(7, 1.0);
  positions[2] = std::numeric_limits<double>::quiet_NaN();
  mock_joint_state_publisher_->publishPositions(positions);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(controller_->update(controller_->get_node()->now(),
                               rclcpp::Duration::from_seconds(0.001)),
            controller_interface::return_type::ERROR);
  for (double command : joint_commands_) EXPECT_DOUBLE_EQ(command, 0.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(rclcpp::ok());
}

TEST_F(JointImpedanceControllerTest, TestPublisherDisconnectStopsWithoutAnotherCallback) {
  controller_->get_node()->set_parameter(rclcpp::Parameter("relative_start", true));
  startController();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  auto period = rclcpp::Duration::from_seconds(0.001);
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::OK);
  mock_joint_state_publisher_->stopPublishing();
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  EXPECT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::ERROR);
  for (double command : joint_commands_) EXPECT_DOUBLE_EQ(command, 0.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(rclcpp::ok());
}

TEST_F(JointImpedanceControllerTest, TestFaultDoesNotWaitForShutdownCleanup) {
  auto context = controller_->get_node()->get_node_base_interface()->get_context();
  const auto handle = context->add_pre_shutdown_callback([]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  });
  startController();
  mock_joint_state_publisher_->stopPublishing();
  mock_joint_state_publisher_->publishPositions(std::vector<double>(7, 1.0), 1.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(controller_->update(controller_->get_node()->now(),
                               rclcpp::Duration::from_seconds(0.001)),
            controller_interface::return_type::ERROR);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(50));
  for (double command : joint_commands_) EXPECT_DOUBLE_EQ(command, 0.0);
  controller_->release_interfaces();
  EXPECT_EQ(controller_->update(controller_->get_node()->now(),
                               rclcpp::Duration::from_seconds(0.001)),
            controller_interface::return_type::ERROR);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_FALSE(rclcpp::ok());
  context->remove_pre_shutdown_callback(handle);
}

TEST_F(JointImpedanceControllerTest, TestRelativeStartRecapturesZeroOnReactivation) {
  controller_->get_node()->set_parameter(rclcpp::Parameter("relative_start", true));
  std::fill(velocity_states_.begin(), velocity_states_.end(), 0.0);
  setRobotPosition(std::vector<double>(7, 0.25));
  startController();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  auto period = rclcpp::Duration::from_seconds(0.001);
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::OK);
  for (double command : joint_commands_) EXPECT_NEAR(command, 0.0, 1e-12);

  // A GELLO delta changes the target before reactivation.
  mock_joint_state_publisher_->stopPublishing();
  mock_joint_state_publisher_->publishPositions(std::vector<double>(7, 1.2));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::OK);
  for (double command : joint_commands_) EXPECT_NEAR(command, 0.2, 1e-9);

  // The same controller instance must take new robot and GELLO zeros after activation.
  setRobotPosition(std::vector<double>(7, 0.75));
  rclcpp_lifecycle::State state;
  ASSERT_EQ(controller_->on_activate(state), CallbackReturn::SUCCESS);
  mock_joint_state_publisher_->publishPositions(std::vector<double>(7, 1.4));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::OK);
  for (double command : joint_commands_) EXPECT_NEAR(command, 0.0, 1e-12);

  mock_joint_state_publisher_->publishPositions(std::vector<double>(7, 1.5));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::OK);
  for (double command : joint_commands_) EXPECT_NEAR(command, 0.1, 1e-9);
}

TEST_F(JointImpedanceControllerTest, TestCommandInterfaceConfiguration) {
  controller_->on_init();
  rclcpp_lifecycle::State state;
  controller_->on_configure(state);

  auto config = controller_->command_interface_configuration();
  EXPECT_EQ(config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  ASSERT_EQ(config.names.size(), command_interfaces_.size());
  for (size_t i = 0; i < command_interfaces_.size(); i++) {
    EXPECT_EQ(config.names[i], joint_names_[i] + "/" + HW_IF_EFFORT);
  }
}

TEST_F(JointImpedanceControllerTest, TestStateInterfaceConfiguration) {
  controller_->on_init();
  rclcpp_lifecycle::State state;
  controller_->on_configure(state);

  auto config = controller_->state_interface_configuration();
  EXPECT_EQ(config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  ASSERT_EQ(config.names.size(), state_interfaces_.size());
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    EXPECT_EQ(config.names[2 * i], joint_names_[i] + "/" + HW_IF_POSITION);
    EXPECT_EQ(config.names[2 * i + 1], joint_names_[i] + "/" + HW_IF_VELOCITY);
  }
}
