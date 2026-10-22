// Copyright (c) 2026
// Licensed under the Apache License, Version 2.0; see LICENSE.
#pragma once

#include <franka/gripper_state.h>

namespace pi05_gripper_driver {

// Injection boundary for tests. Production forwards these calls unchanged to
// one libfranka Gripper; test implementations never construct that connection.
class GripperBackend {
 public:
  virtual ~GripperBackend() = default;
  virtual bool homing() = 0;
  virtual bool move(double width, double speed) = 0;
  virtual bool grasp(double width, double speed, double force, double inner, double outer) = 0;
  virtual bool stop() = 0;
  virtual franka::GripperState readOnce() = 0;
};

}  // namespace pi05_gripper_driver
