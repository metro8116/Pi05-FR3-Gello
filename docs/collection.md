# GELLO / FR3 Data Collection

English · [简体中文](collection_zh.md)

The collection interface runs on the robot computer in this repository's robot Python environment. ROS2 Humble, the GELLO publisher, and the FR3 controllers run in the separate `ros2` container. Opening the interface starts the preview only; the operator must explicitly enable robot control in the interface.

## Configuration and preparation

- `ros2_ws/src/franka_gello_state_publisher/config/my_gello.yaml` retains the GELLO joint directions, offsets, and gripper travel calibration previously used on this computer. This is a historical calibration for the current hardware. Verify and calibrate any other GELLO device separately.
- The default `robot_ip` in `ros2_ws/src/franka_fr3_arm_controllers/config/example_fr3_config.yaml` is `172.16.0.2`. Edit that configuration or set `FRANKA_IP` (the collection entry point also accepts `ROBOT_IP`). The collection launch passes the address to the FR3 driver without rewriting the YAML. For deployment, `robot.ip` in the default `deploy/configs/deploy.yaml` must also match the actual address; select a private copy such as `deploy/configs/local.yaml` through `PI05_CONFIG`.
- Set `GELLO_PORT=/dev/serial/by-id/YOUR_GELLO_DEVICE` to select GELLO explicitly. When it is unset, automatic selection occurs only if exactly one FTDI/OpenRB device is found. Startup refreshes the device link inside the container and verifies the device number without opening or writing to the serial port. Actual GELLO communication begins when the operator starts teleoperation.
- Defaults are `camera_1=233522074047` and `camera_2=405622074065`; override them with `TAKE_DATA_CAMERA_SERIALS=CAMERA_1_SERIAL,CAMERA_2_SERIAL`. Keep the same order and viewpoints for training and deployment. The names do not themselves designate a wrist or exterior camera.
- The current entry point fixes the Franka Hand policy to **Move, 0.04 m/s, with no automatic homing**. Move accepts width and speed parameters, with no grasp-force setting. This configuration must not be interpreted as enforcing a particular force limit.
- The controller retains `relative_start=true`: activation records the current FR3/GELLO angles, then follows GELLO angle changes. It does not automatically move to a recording start pose.

End any model deployment session first, check the Execution/FCI state in Franka Desk, and have the physical stop switch available at the robot. Collection and deployment share a system control lock; the old and new projects are also mutually exclusive. Do not delete the lock file to take control from another session.

Migration neither imports nor clears hardware fault locks from the old project. If that project has an unresolved gripper fault or inspection state, complete the corresponding physical inspection first. The new repository checks `runtime/gripper_hardware_fault.lock` by default; use `FR3_HARDWARE_FAULT_LOCK` to select a host-side lock file. Startup is refused while that file exists. Use `FR3_CONTAINER_HARDWARE_FAULT_LOCK` for a custom container-side lock path; its default is `/workspace/repo/runtime/gripper_hardware_fault.lock`.

## Starting and recording

After configuring this repository's robot environment, ROS image, and workspace, run from the repository root:

```bash
./run.sh collect --output-dir data/raw/avocado_batch01
```

The underlying script can also be called directly:

```bash
bash capture/start_capture.sh --output-dir data/raw/avocado_batch01
```

The default output is this repository's `data/raw/`. Use a separate directory for a new task or batch to avoid mixing historical records into new data. Both `TAKE_DATA_OUTPUT_DIR` and `--output-dir` configure the output path. The default task is `Put the avocado in the basket.`; set `TAKE_DATA_TASK` before startup or edit it in the interface.

Follow this sequence in the GUI:

1. Check both camera images, their order, and the task text. Click Start teleoperation (`启动遥操作`) and wait for the controller and actual measurements to become ready.
2. Arrange the initial scene, then click Start recording (`开始录制`). Recording is refused if the task text is empty or telemetry is missing or stale.
3. Complete the task, retaining the actual gripper opening, placement, and withdrawal. Click Stop recording (`停止录制`) and wait for the files to finish saving.
4. Each episode defaults to `outcome=success`, and the next episode resets to `success`. For a failed or manually interrupted episode, select `failure` or `aborted` before stopping the recording. The default label does not replace the operator's judgment of the actual outcome.
5. Continue with another episode or finish collection.

**Stopping recording stops data saving only; teleoperation continues.** To finish, stop recording, click Stop teleoperation (`停止遥操作`), then close the GUI. Control can also be stopped separately with:

```bash
bash capture/stop_capture.sh
```

The stop entry point reads only this repository's `runs/capture/session.json`, verifies the target container, and stops processes matching the recorded owner and session ID. It does not perform global process-name cleanup. Without session state, it only reports the condition; it does not scan for or kill processes. The stop script does not close the GUI. Close the interface before conversion to ensure the files are no longer being written.

If an abnormal exit left session state behind, a new teleoperation session refuses to start. Run the stop entry point first to verify the old session. Do not bypass the check by deleting its state file.

## Saved data

Each recording is stored in `record_NNN/` under the output directory and contains:

```text
metadata.json                 Authoritative final integrity, task, and outcome state
action/action_data_*.json      Aligned state/action and raw telemetry
action/action_data_*.h5        The same canonical arrays and diagnostic fields
video/camera_1_*.mp4           Camera 1 RGB
video/camera_2_*.mp4           Camera 2 RGB
video/*_frames/frame_*.jpg     Original saved images
video/frame_timestamps.json   Camera serials, exposure/receipt times, and frame numbers
```

Cameras capture at 640×480 and 30 FPS; the target recording rate is 15 FPS. The two cameras do not use hardware-trigger synchronization. Conversion performs additional timing and frame-count checks.

`observation.state` contains seven measured FR3 joint angles followed by the sum of the two measured finger positions. `action` contains seven actual absolute joint targets from the controller followed by the last accepted gripper target width. Joint angles are in rad, widths in m, and both arrays are eight-dimensional float32. GELLO joint readings are retained as diagnostics, not substituted for the robot's action targets.

A gripper Move result indicating an unreached target, or another execution diagnostic, does not establish that an object was grasped or the task succeeded. The task outcome is independent of low-level action results. Data-quality or saving failures still mark the root metadata as `status=error`, even when the outcome defaults to success. Error episodes are retained for diagnosis and excluded by default during conversion.

Physical motion replay is disabled in the new collection interface. Open the MP4 files to review demonstrations.

## Checks without hardware

```bash
bash capture/start_capture.sh --check
python3 tests/test_capture_session.py
```

`--check` only imports Python dependencies. It does not open cameras, call Docker, or connect to the robot. Missing dependencies produce an error and exit; the script does not install pip packages or activate an old conda environment. `CAPTURE_PYTHON` selects the actual interpreter, defaulting to the repository's `.venv-robot/bin/python`.

When dependencies are available, `tests/test_capture_contract.py` verifies the data contract using simulated telemetry and real JSON/HDF5 writes. Environments missing camera/GUI dependencies explicitly skip the test; no hardware is opened.
