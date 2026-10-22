// Copyright (c) 2026
// Licensed under the Apache License, Version 2.0; see LICENSE.
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <pi05_gripper_driver/gripper_action_server.hpp>

using namespace std::chrono_literals;
using pi05_gripper_driver::GripperActionServer;
using pi05_gripper_driver::GripperBackend;

namespace {
template <class Predicate>
bool waitUntil(Predicate predicate, std::chrono::milliseconds timeout = 2500ms) {
  const auto end = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < end) {
    if (predicate()) return true;
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

class FakeBackend final : public GripperBackend {
 public:
  bool homing() override { return false; }
  bool grasp(double, double, double, double, double) override { return false; }
  bool move(double width, double speed) override {
    std::lock_guard<std::mutex> guard(mutex);
    moved_width = width;
    moved_speed = speed;
    return true;
  }
  bool stop() override {
    std::unique_lock<std::mutex> guard(mutex);
    ++stop_calls;
    ++active_stops;
    max_active_stops = std::max(max_active_stops, active_stops);
    condition.wait(guard, [this] { return release_stop; });
    --active_stops;
    return stop_success;
  }
  franka::GripperState readOnce() override {
    if (fail_read.load()) {
      ++failed_reads;
      throw franka::NetworkException("injected read failure; no real gripper exists");
    }
    franka::GripperState state{};
    state.width = .04 + (++reads) * .000001;
    state.max_width = .08;
    return state;
  }
  void release() {
    std::lock_guard<std::mutex> guard(mutex);
    release_stop = true;
    condition.notify_all();
  }
  int stopConcurrency() {
    std::lock_guard<std::mutex> guard(mutex);
    return max_active_stops;
  }
  std::atomic<int> reads{0}, failed_reads{0}, stop_calls{0};
  std::atomic<bool> fail_read{false};
  std::mutex mutex;
  std::condition_variable condition;
  bool release_stop{false}, stop_success{true};
  int active_stops{0}, max_active_stops{0};
  double moved_width{0.}, moved_speed{0.};
};

class DriverTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
  void SetUp() override {
    static int sequence = 0;
    const std::string name = "pi05_driver_test_" + std::to_string(++sequence);
    backend = std::make_shared<FakeBackend>();
    rclcpp::NodeOptions options;
    options.arguments({"--ros-args", "-r", "__node:=" + name});
    options.parameter_overrides({rclcpp::Parameter("robot_ip", "unused-injected-backend"),
        rclcpp::Parameter("joint_names", std::vector<std::string>{"fr3_finger_joint1", "fr3_finger_joint2"})});
    driver = std::make_shared<GripperActionServer>(options, backend);
    observer = std::make_shared<rclcpp::Node>(name + "_observer");
    subscription = observer->create_subscription<sensor_msgs::msg::JointState>(
        "/" + name + "/joint_states", 100, [this](sensor_msgs::msg::JointState::SharedPtr message) {
          std::lock_guard<std::mutex> guard(samples_mutex);
          samples.push_back(*message);
        });
    stop = observer->create_client<std_srvs::srv::Trigger>("/" + name + "/stop");
    move = rclcpp_action::create_client<franka_msgs::action::Move>(observer, "/" + name + "/move");
    executor = std::make_unique<rclcpp::executors::MultiThreadedExecutor>(rclcpp::ExecutorOptions{}, 2);
    executor->add_node(driver);
    executor->add_node(observer);
    worker = std::thread([this] { executor->spin(); });
    ASSERT_TRUE(stop->wait_for_service(2s));
    ASSERT_TRUE(waitUntil([this] { return sampleCount() >= 3; }));
  }
  void TearDown() override {
    backend->release();
    executor->cancel();
    if (worker.joinable()) worker.join();
    executor->remove_node(driver);
    executor->remove_node(observer);
  }
  std::size_t sampleCount() {
    std::lock_guard<std::mutex> guard(samples_mutex);
    return samples.size();
  }
  std::vector<sensor_msgs::msg::JointState> snapshot() {
    std::lock_guard<std::mutex> guard(samples_mutex);
    return samples;
  }
  std::shared_ptr<FakeBackend> backend;
  std::shared_ptr<GripperActionServer> driver;
  rclcpp::Node::SharedPtr observer;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr stop;
  rclcpp_action::Client<franka_msgs::action::Move>::SharedPtr move;
  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor;
  std::thread worker;
  std::mutex samples_mutex;
  std::vector<sensor_msgs::msg::JointState> samples;
};

TEST_F(DriverTest, BlockingStopDoesNotStopFreshStateAndTwoStopsStaySerialized) {
  auto first = stop->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  ASSERT_TRUE(waitUntil([this] { return backend->stop_calls.load() == 1; }));
  const auto before = sampleCount();
  auto second = stop->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  // Longer than the live gripper freshness limit: a fake blocking Stop must not
  // prevent genuine new reads/publications from the independent callback group.
  std::this_thread::sleep_for(650ms);
  EXPECT_GE(sampleCount(), before + 10);
  EXPECT_EQ(backend->stop_calls.load(), 1);
  EXPECT_EQ(first.wait_for(0ms), std::future_status::timeout);
  auto states = snapshot();
  ASSERT_GE(states.size(), before + 10);
  EXPECT_GT(rclcpp::Time(states.back().header.stamp).nanoseconds(),
            rclcpp::Time(states[before - 1].header.stamp).nanoseconds());
  EXPECT_GT(states.back().position[0], states[before - 1].position[0]);
  EXPECT_EQ(states.back().name, (std::vector<std::string>{"fr3_finger_joint1", "fr3_finger_joint2"}));
  backend->release();
  ASSERT_EQ(first.wait_for(2s), std::future_status::ready);
  ASSERT_EQ(second.wait_for(2s), std::future_status::ready);
  EXPECT_TRUE(first.get()->success);
  EXPECT_TRUE(second.get()->success);
  EXPECT_EQ(backend->stopConcurrency(), 1);
}

TEST_F(DriverTest, StopFailureIsReportedAsFailureWhileStatesRemainAvailable) {
  backend->stop_success = false;
  backend->release();
  auto response = stop->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  ASSERT_EQ(response.wait_for(2s), std::future_status::ready);
  EXPECT_FALSE(response.get()->success);
  const auto before = sampleCount();
  EXPECT_TRUE(waitUntil([this, before] { return sampleCount() > before; }));
}

TEST_F(DriverTest, ReadFailureDoesNotRepublishOldWidthWithFreshTimestamp) {
  backend->fail_read = true;
  ASSERT_TRUE(waitUntil([this] { return backend->failed_reads.load() >= 3; }));
  const auto before = sampleCount();
  const auto last = snapshot().back().header.stamp;
  std::this_thread::sleep_for(150ms);
  EXPECT_EQ(sampleCount(), before);
  EXPECT_EQ(snapshot().back().header.stamp, last);
  backend->fail_read = false;
  EXPECT_TRUE(waitUntil([this, before] { return sampleCount() > before; }));
}

TEST_F(DriverTest, MoveWidthSpeedAndRawSuccessPassThroughUnchanged) {
  ASSERT_TRUE(move->wait_for_action_server(2s));
  franka_msgs::action::Move::Goal goal;
  goal.width = .066;
  goal.speed = .04;
  auto acceptance = move->async_send_goal(goal);
  ASSERT_EQ(acceptance.wait_for(2s), std::future_status::ready);
  auto handle = acceptance.get();
  ASSERT_NE(handle, nullptr);
  auto terminal = move->async_get_result(handle);
  ASSERT_EQ(terminal.wait_for(2s), std::future_status::ready);
  const auto result = terminal.get();
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_TRUE(result.result->success);
  EXPECT_TRUE(result.result->error.empty());
  EXPECT_DOUBLE_EQ(backend->moved_width, .066);
  EXPECT_DOUBLE_EQ(backend->moved_speed, .04);
  EXPECT_EQ(backend->stop_calls.load(), 0);
}
}  // namespace
