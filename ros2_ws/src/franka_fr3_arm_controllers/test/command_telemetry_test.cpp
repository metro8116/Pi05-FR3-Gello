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

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sstream>
#include <std_msgs/msg/string.hpp>

class CommandTelemetryTest : public JointImpedanceControllerTest {
 protected:
  void SetUp() override {
    JointImpedanceControllerTest::SetUp();
    subscription_ = controller_->get_node()->create_subscription<std_msgs::msg::String>(
        "/gello_capture/arm_command", rclcpp::QoS(10),
        [this](const std_msgs::msg::String& message) {
          std::lock_guard<std::mutex> lock(messages_mutex_);
          messages_.push_back(message.data);
          message_cv_.notify_all();
        });
  }

  bool waitForMessages(size_t count) {
    std::unique_lock<std::mutex> lock(messages_mutex_);
    return message_cv_.wait_for(lock, std::chrono::seconds(1),
                                [this, count]() { return messages_.size() >= count; });
  }

  size_t messageCount() {
    std::lock_guard<std::mutex> lock(messages_mutex_);
    return messages_.size();
  }

  boost::property_tree::ptree message(size_t index) {
    std::string data;
    {
      std::lock_guard<std::mutex> lock(messages_mutex_);
      data = messages_.at(index);
    }
    boost::property_tree::ptree parsed;
    std::istringstream stream(data);
    boost::property_tree::read_json(stream, parsed);
    return parsed;
  }

  std::vector<double> array(const boost::property_tree::ptree& json, const char* name) {
    std::vector<double> values;
    for (const auto& item : json.get_child(name)) values.push_back(item.second.get_value<double>());
    EXPECT_EQ(values.size(), 7u);
    return values;
  }

  void expectArray(const boost::property_tree::ptree& json, const char* name,
                   const std::vector<double>& expected) {
    const auto actual = array(json, name);
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-12);
  }

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  std::mutex messages_mutex_;
  std::condition_variable message_cv_;
  std::vector<std::string> messages_;
};

TEST_F(CommandTelemetryTest, NoCommandIsInventedBeforeInitialization) {
  controller_->on_init();
  rclcpp_lifecycle::State state;
  ASSERT_EQ(controller_->on_configure(state), CallbackReturn::SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  EXPECT_EQ(messageCount(), 0u);
  mock_joint_state_publisher_->stopPublishing();
  ASSERT_EQ(controller_->on_activate(state), CallbackReturn::SUCCESS);
  ASSERT_EQ(controller_->update(controller_->get_node()->now(),
                                 rclcpp::Duration::from_seconds(0.001)),
            controller_interface::return_type::OK);
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  EXPECT_EQ(messageCount(), 0u);
}

TEST_F(CommandTelemetryTest, RelativeTargetAndMeasuredStateComeFromSameUpdate) {
  controller_->get_node()->set_parameter(rclcpp::Parameter("relative_start", true));
  const std::vector<double> start_q{0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7};
  const std::vector<double> initial_dq{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
  setRobotPosition(start_q);
  velocity_states_ = initial_dq;
  ASSERT_EQ(startController(), CallbackReturn::SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const auto period = rclcpp::Duration::from_seconds(0.001);
  const int64_t first_stamp = 1792530000123456789LL;
  ASSERT_EQ(controller_->update(rclcpp::Time(first_stamp, RCL_ROS_TIME), period),
            controller_interface::return_type::OK);
  ASSERT_TRUE(waitForMessages(1));
  const auto first = message(0);
  EXPECT_EQ(first.get<int>("schema_version"), 1);
  EXPECT_EQ(first.get<uint64_t>("sequence"), 1u);
  EXPECT_EQ(first.get<int64_t>("stamp_ns"), first_stamp);
  EXPECT_GT(std::stoll(first.get<std::string>("activation_id")), 0);
  EXPECT_TRUE(first.get<bool>("relative_start"));
  std::vector<std::string> names;
  for (const auto& item : first.get_child("joint_names")) {
    names.push_back(item.second.get_value<std::string>());
  }
  EXPECT_EQ(names, std::vector<std::string>(joint_names_.begin(), joint_names_.end()));
  expectArray(first, "q_measured", start_q);
  expectArray(first, "dq_measured", initial_dq);
  expectArray(first, "q_target", start_q);
  expectArray(first, "robot_start_q", start_q);
  expectArray(first, "gello_start_q", std::vector<double>(7, 1.0));
  expectArray(first, "tau_command", joint_commands_);

  mock_joint_state_publisher_->stopPublishing();
  const std::vector<double> gello_q{1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
  mock_joint_state_publisher_->publishPositions(gello_q);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const std::vector<double> measured_q{0.15, 0.25, 0.35, -0.35, 0.55, 0.65, 0.75};
  const std::vector<double> measured_dq{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  setRobotPosition(measured_q);
  velocity_states_ = measured_dq;
  const int64_t second_stamp = first_stamp + 1000000;
  ASSERT_EQ(controller_->update(rclcpp::Time(second_stamp, RCL_ROS_TIME), period),
            controller_interface::return_type::OK);
  const auto sent_torque = joint_commands_;
  // Mutating the mock hardware before the timer runs must not alter that cycle's snapshot.
  setRobotPosition(std::vector<double>(7, 99.0));
  velocity_states_.assign(7, 99.0);
  ASSERT_TRUE(waitForMessages(2));
  const auto second = message(1);
  EXPECT_EQ(second.get<uint64_t>("sequence"), 2u);
  EXPECT_EQ(second.get<int64_t>("stamp_ns"), second_stamp);
  EXPECT_EQ(second.get<std::string>("activation_id"), first.get<std::string>("activation_id"));
  std::vector<double> target(7);
  for (size_t i = 0; i < target.size(); ++i) {
    target[i] = start_q[i] + gello_q[i] - 1.0;
    const double filtered_velocity = 0.25 * initial_dq[i] + 0.5 * measured_dq[i];
    EXPECT_NEAR(sent_torque[i], target[i] - measured_q[i] - 0.1 * filtered_velocity, 1e-12);
  }
  expectArray(second, "q_target", target);
  expectArray(second, "q_measured", measured_q);
  expectArray(second, "dq_measured", measured_dq);
  expectArray(second, "tau_command", sent_torque);
  expectArray(second, "gello_q", gello_q);
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  EXPECT_EQ(messageCount(), 2u);  // A timer tick without a new update is not a new sample.
}

TEST_F(CommandTelemetryTest, ReactivationChangesIdAndResetsSnapshotAndSequence) {
  controller_->get_node()->set_parameter(rclcpp::Parameter("relative_start", true));
  std::fill(velocity_states_.begin(), velocity_states_.end(), 0.0);
  setRobotPosition(std::vector<double>(7, 0.25));
  ASSERT_EQ(startController(), CallbackReturn::SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const auto period = rclcpp::Duration::from_seconds(0.001);
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::OK);
  ASSERT_TRUE(waitForMessages(1));
  const auto first_id = message(0).get<std::string>("activation_id");
  rclcpp_lifecycle::State state;
  ASSERT_EQ(controller_->on_deactivate(state), CallbackReturn::SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(messageCount(), 1u);
  setRobotPosition(std::vector<double>(7, 0.75));
  ASSERT_EQ(controller_->on_activate(state), CallbackReturn::SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(messageCount(), 1u);
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::OK);
  ASSERT_TRUE(waitForMessages(2));
  const auto second = message(1);
  EXPECT_NE(second.get<std::string>("activation_id"), first_id);
  EXPECT_EQ(second.get<uint64_t>("sequence"), 1u);
  expectArray(second, "q_target", std::vector<double>(7, 0.75));
}

TEST_F(CommandTelemetryTest, FaultDoesNotPublishANewCommandSnapshot) {
  controller_->get_node()->set_parameter(rclcpp::Parameter("relative_start", true));
  ASSERT_EQ(startController(), CallbackReturn::SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const auto period = rclcpp::Duration::from_seconds(0.001);
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::OK);
  ASSERT_TRUE(waitForMessages(1));
  mock_joint_state_publisher_->stopPublishing();
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  ASSERT_EQ(controller_->update(controller_->get_node()->now(), period),
            controller_interface::return_type::ERROR);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(messageCount(), 1u);
}
