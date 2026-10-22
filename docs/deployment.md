# FR3 / PI05 Deployment Across Two Machines

English · [简体中文](deployment_zh.md)

The robot computer runs the cameras, GUI, ROS control bridge, and robot connection; the cloud GPU server runs model inference. Both machines use the same version of this repository. Select models and datasets through each machine's configuration or environment variables. A separate training project is not required. Keep data, checkpoints, and private SSH configuration out of Git.

The default deployment configuration is `deploy/configs/deploy.yaml`. Set `PI05_CONFIG` to use your own configuration copy. `run.sh` reads `configs/local.env` and `configs/autodl.env`; see the root README for creating these files and selecting Python environments. Run all commands below from the repository root, using your own SSH address and login port.

## 1. Start the inference service on the cloud GPU server

Select an exported official `pretrained_model` checkpoint. The service loads its model configuration, native weights, saved pre/postprocessors, and statistics. It does not recompute statistics from training data. Training and deployment use the same `fr3_pi05.runtime.load_native_weights_strict` directly; native checkpoints do not pass through the base-model key remapping.

```bash
./run.sh serve --checkpoint checkpoints/SELECTED_CHECKPOINT/pretrained_model \
  --host 127.0.0.1 --port 8000 --device cuda --dtype bfloat16
```

Alternatively, set `PI05_CHECKPOINT` in the cloud GPU server's environment file, then run `./run.sh serve`. `--train-project` defaults to this repository's root, so no separate training project needs to be supplied.

To inspect the local checkpoint and runtime first:

```bash
./run.sh serve --checkpoint checkpoints/SELECTED_CHECKPOINT/pretrained_model --check-only
```

`--check-only` does not allocate the large PI05 model, but it still checks the selected device and dependencies. The real service performs one synthetic-input warmup by default. It neither reads the training dataset nor connects to the robot. It binds only to the server's loopback address and is reached through SSH forwarding. Checkpoint symlinks such as `last` are resolved when loading; later training saves do not automatically switch an already loaded service to new weights.

To check only the communication path:

```bash
./run.sh serve --mock --warmup 0 --device cpu
```

Mock metadata explicitly includes `mock=true`. The robot computer's supervisor prevents a mock service from starting the robot control stack.

## 2. Configure SSH forwarding on the robot computer

Set the following in the private `configs/autodl.env` file on the robot computer. Keep `export` so the supervisor receives these variables:

```bash
export PI05_SSH_HOST="YOUR_CLOUD_GPU_HOST"
export PI05_SSH_USER="YOUR_SSH_USER"
export PI05_SSH_PORT="YOUR_SSH_PORT"
export PI05_LOCAL_PORT="18000"
export PI05_REMOTE_PORT="8000"
```

Replace `YOUR_...` with the actual settings. Verify the SSH host key and configure key-based login first. The supervisor uses `BatchMode=yes` and `StrictHostKeyChecking=yes`; it will not wait in the background for a password or first-time host-key confirmation.

With a normal `./run.sh deploy` invocation, the supervisor manages its own SSH tunnel. To keep a tunnel separately, establish it in another terminal on the robot computer:

```bash
source configs/autodl.env
ssh -N -T -p "$PI05_SSH_PORT" \
  -L "127.0.0.1:18000:127.0.0.1:8000" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
  -o BatchMode=yes -o StrictHostKeyChecking=yes \
  "$PI05_SSH_USER@$PI05_SSH_HOST"
```

Then use `./run.sh deploy --no-tunnel` to avoid binding the same local port twice. The local forwarding port must match `model.port` in the deployment configuration; the server port must match `serve --port`. The example values `18000` and `8000` are software defaults, not SSH login ports.

To check only the tunnel and service metadata, without starting cameras, ROS, or the robot:

```bash
./run.sh deploy --network-check
# If a tunnel has already been established manually:
./run.sh deploy --network-check --no-tunnel
```

## 3. Prepare the local ROS overlay

Prepare the new ROS workspace and container as described in the root README. The default container name is `pi05-fr3-gello-ros2-1`, and this repository's `ros2_ws/` is mounted at `/workspace/ros2` inside it. Then run:

```bash
./run.sh prepare
./run.sh deploy --check
```

`prepare` builds the separate `pi05_deploy_overlay` and runs the controller and gripper isolation tests without starting robot nodes. `--check` checks the configuration, camera Python dependencies, and successful overlay-build marker without connecting to the robot. Run prepare again after changes to controller or gripper source code. A missing driver binary or test-success marker prevents startup.

The separate gripper package is in `ros2_ws/src/pi05_gripper_driver`, with its LICENSE and NOTICE retained. It uses an independent Stop path and state publication, without replacing the system-installed `franka_gripper` package.

## 4. GUI operation sequence

End collection, GELLO teleoperation, and any other control session before running:

```bash
./run.sh deploy
# Or use an existing manually established tunnel:
./run.sh deploy --no-tunnel
```

The supervisor verifies model metadata, starts this session's ROS bundle, and waits for the bridge and actual state to become ready before opening the GUI. The impedance controller initially remains **inactive**. Opening the window does not automatically enable model control.

1. Click Connect and preview (`连接并预览`). This connects the model, actual state, and two cameras without sending model actions.
2. Optionally click Predict only, no motion (`仅预测（不运动）`). It performs one inference on the current images and actual state, logging the prediction and deviations in `client.log`. It neither submits actions nor enables the controller.
3. To release an object, click Open gripper, 80 mm (`打开夹爪（80 mm）`) while in preview with the model disabled, the arm stationary, and all Move/Stop operations complete. The target is a total opening of 80 mm at 0.04 m/s; the arm controller remains inactive. The GUI reports completion only after the action succeeds and a fresh measured opening reaches at least 79 mm.
4. Click Enable model execution (`启用模型执行`) and confirm the physical conditions to create a new execution session. The bridge checks the actual pose, state freshness, model identity, and controller state, and executes only after activation is confirmed.
5. Click Stop model execution (`停止模型执行`) to revoke the session immediately and send an independent stop request. The window and preview remain available. Before running again, wait for stop cleanup to complete and explicitly enable execution again.

Manual gripper opening, prediction-only mode, and enabling execution cannot run concurrently. Server predictions do not reach robot control unless execution has been explicitly enabled.

### Automatic execution recording

When model execution is enabled, recording of both real RGB streams starts first. The system verifies both encoders and the first writable frames before sending arm. Stopping model execution, a protective disable, a recording error, or closing the window immediately revokes execution and stops accepting new recording frames; frames already received are saved in the background. Preview, prediction-only mode, and separate manual gripper opening do not start recording. Execution recording continues while waiting for the native Stop handshake.

The default directory is `deploy/recordings/<time>_<session>_<unique>/`. Each enable operation creates a new directory containing `camera_1.mp4`, `camera_2.mp4`, `frames.jsonl`, `metadata.json`, and encoder logs. The old deployment project uses its own `recordings/` directory. Override the output root with `PI05_RECORDING_DIR`; relative paths are resolved against the deployment directory. The GUI shows recording (`录像中`), saving (`保存中`), or saved (`已保存`) status, the path, frame counts, and dropped-frame counts.

Recording saves actual 640×480 camera RGB frames, encoded as H.264 on the CPU at a nominal 30 FPS; it does not capture the GUI preview. Metadata includes camera serials, FPS, start/stop-request/save-completion times, the stop reason, and per-camera frame and drop counts. The per-frame journal records source timestamps and encoded frame indices.

The writer uses an asynchronous bounded queue, defaulting to `recording.queue_size=64`. Camera and stop-control threads do not wait for disk writes. Drops due to a full queue or bounded finalization are recorded explicitly rather than hidden through duplicated images. MP4 uses a nominal frame rate, so consult the timestamp journal when interpreting duration if frames were dropped. Re-enabling is blocked while the previous recording is being saved. On a supervisor shutdown notification, the GUI saves first, then any remaining encoders are cleaned up.

An existing FFmpeg executable is required. Resolution checks, in order, `PI05_FFMPEG`, the system command, the current Python environment's `imageio_ffmpeg`, and an existing binary in an explicitly selected `PI05_CONVERTER_PYTHON` environment. It does not download or install anything. The new robot profile includes `imageio-ffmpeg`; when reusing an old environment, `PI05_FFMPEG` can point to an existing executable. A recording startup failure prevents arming; a recording failure during execution reports an error and disables the model. This feature requires only reopening the local deployment window, without rebuilding the ROS overlay or restarting the cloud GPU server's service. Recordings remain separate from training data and are not automatically added to the dataset.

## 5. Default action interface and gates

The following values describe the repository's current source and default configuration. They constrain different stages; one timeout must not be treated as a substitute for another.

| Item | Current value or behavior |
| --- | --- |
| Task | `Put the avocado in the basket.` |
| State and action | Eight dimensions each: state contains seven measured joint angles in rad plus measured total gripper width in m; action contains seven absolute joint-angle targets in rad plus target total width in m |
| Model | 15 Hz, returning 50 steps per prediction; executes only the aligned prefix of at most six steps, about 0.4 s |
| Local interpolation | 100 Hz; the underlying controller update rate is 1000 Hz |
| Initial target difference / adjacent waypoint difference | Maximum 0.4 rad / 0.15 rad |
| Tracking error / joint velocity / acceleration | Maximum 0.25 rad / 0.35 rad/s / 0.8 rad/s² |
| Gripper | Native `Move`, 0.04 m/s, total opening 0–0.08 m, deadband 0.002 m |
| Move acceptance/execution timeout | 5.0 s; the overall manual-opening polling deadline is 6.0 s |
| Native Stop / opening-preemption handshake deadline | 3.0 s |
| Controller deactivation confirmation deadline | 1.2 s |
| Heartbeat / action submission timeout | 0.5 s / 0.8 s |
| Maximum observation age | 1.0 s; this does not extend the action submission timeout |
| Maximum measured joint / gripper state age | 0.2 s / 0.5 s |
| Cameras | Fixed two-camera order, 640×480 RGB at 30 FPS; maximum frame age 0.2 s and inter-camera skew 0.08 s |

Shape, numerical validity, and joint limits are checked for all 50 output steps, not just the prefix about to execute. Small model gripper-range violations retain the raw values in the log and are clamped to the physical range. Gross violations, invalid measured state, and discontinuous joint targets are still rejected.

### Switching the gripper from closing to opening

If an accepted closing Move is still running when a fresh model target requests opening, the bridge uses a single native Stop path. It does not also send an action cancellation, avoiding overlapping Stop calls.

The GUI displays Waiting for the gripper to stop; the arm is decelerating to a hold (`等待夹爪停止，机械臂正在减速保持`). During the wait, joint targets decelerate under the existing velocity and acceleration limits, then hold. Joint waypoints received during this stage are not replayed after the handshake finishes. State freshness, heartbeat, tracking-error checks, and the other watchdogs remain active.

The bridge requires both a successful native Stop and a terminal result for the old Move before treating the old target as finished. It then requires a fresh gripper measurement captured after stop confirmation and an opening target that is still fresh. It never replays cached targets from an asynchronous callback. After opening is released, resuming the arm trajectory also requires a subsequent fresh submission. The three-second bound is the stop-handshake deadline, not a grace period that suppresses other faults.

## 6. Stopping, fault windows, and restarting

Closing the GUI revokes the current execution. The supervisor then confirms that all ROS descendants belonging to the session have exited and closes its SSH tunnel. The session can also be stopped from another terminal on the robot computer:

```bash
./run.sh stop-deploy
```

Cleanup verifies the current bundle and PID/start-time identities and affects only processes owned by this session. If complete ROS exit has not been confirmed, the control lease and recovery journal are retained while cleanup retries. Sending a stop request is not treated as proof that stopping has finished.

If ROS exits, a fatal gripper fault occurs, the tunnel exits, or the bridge remains unreachable during operation, the supervisor first stops ROS and the tunnel but **keeps the GUI open** to show the fault and log paths. The fault window revokes its token, disables reconnect/predict/open/enable controls, and stops the cameras. Late RPC replies cannot restore an old ready or executing state. The fault window never automatically rebuilds ROS or re-arms the robot.

The first meaningful ROS FATAL/ERROR and its original source line are written to this session's `session_status.json`; later `Connection refused` or `Bridge process exiting` messages do not mask it. The window reports ROS and the tunnel as stopped only after their cleanup is confirmed.

While the window is retained, the control lease and journal remain in place, preventing another deployment or collection session from taking the resources. Close the fault window or run `stop-deploy` to finish exiting, then inspect the physical state before restarting. The stop script also works for a fault window.

The local supervisor additionally probes the bridge state RPC every 0.5 seconds, with a 0.3-second timeout per request. Two seconds without a valid response triggers this session's cleanup path. A responsive RPC reporting only `ready=false` or an ordinary fault is not treated as a lost process.

## 7. Logs and migration verification

Each session writes to `deploy/logs/<session>/`:

- `ros.log`: ROS, controller, and gripper lifecycle. Gripper events include targets, measured opening, and raw action results.
- `ssh.log`: the SSH tunnel managed by the supervisor.
- `client.log`: connections, prediction-only results, model requests, and GUI errors.
- `session_status.json`: the first cause, source line, and cleanup stage saved when a runtime fault occurs.

The deployment migration copied only source, configuration, and tests. Old logs, runs, data, private environment files, and verification reports were not copied into the new repository. The new deployment uses this repository's shared control-lock implementation and training native loader. The lock remains `/tmp/fr3-control-<uid>.lock`, preserving mutual exclusion with old sessions.

The deployment migration verification did not start new Docker/ROS control sessions, deployment SSH tunnels, the robot, or GPU workloads. Dependency work separately inspected the existing cloud GPU server environment in read-only mode. Deployment CPU regression commands were:

```bash
cd deploy
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tests -p test_repository_layout.py -v
```

The migrated suite contained 220 tests. With the default Python interpreter, 208 passed and 12 were skipped: 11 tiny CPU torch/safetensors native-loading tests and one official CPU processor integration test, because the interpreter lacked their dependencies. Four additional repository-layout tests all passed, covering the new default paths/container, shared implementation identity, the checkpoint environment variable, and a simulated packaged bridge displaying help without the host repository's `PYTHONPATH`. The initial migration therefore contained 224 tests.

After automatic recording was added, the complete regression run contained 252 tests in the new repository: 239 passed and 13 were skipped. The existing deployment project contained 248 tests: 235 passed and 13 were skipped. In addition to the 12 ML dependency skips above, the default Python lacked cv2, so the real MP4 decode test was skipped. Both versions of `test_execution_video.py` were then rerun with the existing robot Python, and all 17 tests in each passed. These included actual H.264 MP4 encoding and OpenCV decoding of synthetic dual-camera RGB frames; no real camera was used. The added GUI save-grace test used harmless subprocesses to confirm that encoders did not receive TERM before a normal save completed.

Process-lifecycle tests create only harmless Linux subprocesses owned by the test, using a fake `ros2` to verify orphan adoption and INT/TERM/KILL cleanup. Network tests use only local test ports. These results do not establish verification of real GPU inference, ROS driver compilation, or physical robot operation. Before actual deployment, rerun the skipped CPU tests in the appropriate ML environment, then separately check the scene and actual hardware state.
