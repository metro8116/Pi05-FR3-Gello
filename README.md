<h1 align="center">Pi05 × FR3 × GELLO</h1>

<h3 align="center">From GELLO Demonstrations to π0.5 Deployment on a Real Robot</h3>

<p align="center">
  English · <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/Metroooooooo/pi05-fr3-gello-avocado-12k">Model</a> /
  <a href="https://huggingface.co/datasets/Metroooooooo/fr3-gello-avocado-v30">Dataset</a> ·
  🎮 <a href="https://github.com/metro8116/Pi05-Mujoco">Simulation project</a>
</p>

<a id="overview"></a>

## 📌 Overview

A complete workflow for **GELLO, a Franka FR3 arm with a parallel gripper, and two RealSense cameras**: collect demonstrations, convert the recordings, fine-tune [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base), and deploy the policy on the real robot for the instruction:

> **Put the avocado in the basket.**

The robot PC handles cameras, user interfaces, and ROS2 real-time control. A cloud GPU server handles π0.5 training and inference. **Use the same repository on both machines, with different commands for each role.**

- **Demonstration collection:** GELLO teleoperation, two camera previews, joint plots, episode recording, and outcome labels.
- **Data processing:** raw recording validation, reviewed trimming plans, LeRobot Dataset **v3.0** export, and readback checks.
- **Fine-tuning:** the **official LeRobot 0.6.1 trainer**, retaining its sampling, Accelerate integration, optimizer, and checkpoint save/resume mechanisms.
- **Robot deployment:** cloud GPU inference, SSH communication, a local deployment GUI, standalone gripper opening, and execution stop controls.

<a id="contents"></a>

## 📚 Contents

- [🛠️ Installation](#-installation)
- [🎮 Collect demonstrations](#-collect-demonstrations)
- [🎬 Review and convert data](#-review-and-convert-data)
- [🧠 Train and resume π0.5](#-train-and-resume-π05)
- [🤗 Models and datasets](#-models-and-datasets)
- [🚀 Deploy π0.5](#-deploy-π05)
- [📈 Validation scope](#-validation-scope)
- [📖 Documentation](#-documentation)
- [🙏 Acknowledgements and licensing](#-acknowledgements-and-licensing)

<details>
<summary>📂 Project structure</summary>

```text
Pi05-FR3-Gello/
├── run.sh                          # Unified collection, conversion, training, and deployment entry point
├── capture/                        # Camera GUI, telemetry bridge, and teleoperation stop management
├── tools/convert_to_lerobot_v30.py  # Raw recording validation and v3.0 conversion
├── train.py                        # Adapter for the official LeRobot trainer
├── check_env.py                    # Dataset, model, and ML environment checks
├── src/fr3_pi05/                   # FR3 data contract, processors, and weight loading
├── deploy/                         # Inference server, local GUI, and control bridge
├── common/                         # Control ownership shared by collection and deployment
├── ros2_ws/src/                    # GELLO publisher, FR3 controller, and gripper drivers
├── docker/                         # ROS Humble image and pinned source preparation
├── .devcontainer/                  # VS Code Dev Container configuration
├── requirements/                   # robot / converter / ml profiles and dependency locks
├── configs/                        # Robot PC and cloud GPU configuration templates
├── docs/                           # Setup, collection, data, training, and deployment guides
├── tests/                          # Hardware-free regression and synthetic data tests
├── data/                           # Raw recordings and v3.0 datasets, created at runtime
└── outputs/, runs/, runtime/        # Training outputs, logs, and session state
```

Datasets, weights, virtual environments, videos, and private configuration are ignored by Git and transferred separately. Deployment logs go to `deploy/logs/`; execution videos go to `deploy/recordings/`.

</details>

<a id="installation"></a>

## 🛠️ Installation

### 💻 Machines, hardware, and environments

| Location | Environment | Purpose |
| --- | --- | --- |
| Robot PC | Ubuntu 20.04 / 22.04 with Franka-compatible real-time kernel and networking | Connect the FR3, GELLO, and two USB 3 RealSense cameras |
| Robot PC | `.venv-robot`, Python **3.10** | Collection GUI, cameras, and deployment client |
| Robot PC | ROS2 **Humble** Docker / VS Code Dev Container | GELLO, arm controller, gripper, and ROS bridge |
| Data processing machine | `.venv-converter`, Python **3.12** | LeRobot **0.6.1**, CPU PyTorch **2.11.0**, and dataset conversion |
| Cloud GPU server | `.venv-ml`, Python **3.12** | LeRobot **0.6.1**, PyTorch **2.11.0+cu128**, training and inference |

The three Python environments are created separately with **uv**. `run.sh` selects the interpreter, so manual Conda activation is unnecessary. Conversion and training use the same LeRobot version with CPU and CUDA dependencies, respectively. **Dataset v3.0 refers to the on-disk data format.**

For a new machine, first complete the [system, real-time kernel, Docker, uv, USB permissions, and FR3 network setup](docs/environments.md). The ROS container shares the host kernel; the host must still meet Franka's real-time requirements.

### 🤖 Robot PC: collection and control

Place the repository on the robot PC, for example at `~/桌面/Pi05-FR3-Gello` on the current machine, and run these commands from its root:

```bash
cd ~/桌面/Pi05-FR3-Gello
cp -n configs/local.env.example configs/local.env
# Edit configs/local.env: verify the FR3 IP, GELLO serial port, and both camera serial numbers.

./run.sh setup robot
./run.sh setup converter
./run.sh ros-image
./run.sh ros-up
./run.sh ros-build
./run.sh doctor
./run.sh collect --check
```

`ros-image` builds the dedicated image from pinned source commits with SHA256 verification, reusing completed Docker layers and source downloads. `ros-up` starts an idle container; `ros-build` compiles the collection packages. These three commands do not initiate robot motion.

Skip `setup` for an environment that is already configured. After an interrupted installation, use `./run.sh setup converter --resume`, substituting the appropriate profile when needed. This resumes environment installation, separately from model checkpoint recovery.

The default FR3 IP is `172.16.0.2`. The GELLO calibration and camera configuration belong to the current hardware. When changing hardware, follow the [setup guide](docs/environments.md) to check joint signs, offsets, and camera order. In VS Code, select **Dev Containers: Reopen in Container** for ROS development. Collection and deployment GUIs run on the host.

### ☁️ Cloud GPU: training and inference

Copy the same repository version to the cloud GPU server, for example `/root/autodl-tmp/Pi05-FR3-Gello`:

```bash
cd /root/autodl-tmp/Pi05-FR3-Gello
cp -n configs/autodl.env.example configs/autodl.env

# Run only when creating a new ML environment.
./run.sh setup ml
```

To reuse a working uv environment, skip `setup ml` and set its interpreter and asset paths in `configs/autodl.env`:

```bash
export PI05_ML_PYTHON="/absolute/path/to/existing/environment/bin/python"
export PI05_BASE_MODEL="/absolute/path/models--lerobot--pi05_base/snapshots/YOUR_SNAPSHOT_HASH"
export PI05_TOKENIZER="google/paligemma-3b-pt-224"
```

`run.sh` reads the configuration automatically. The base model path must contain `config.json` and one complete `model.safetensors`. Select the specific snapshot directory rather than the outer Hub cache directory. The tokenizer can use an existing cached ID or a complete local directory.

<details>
<summary>📦 Download the base model and tokenizer for the first time</summary>

With a newly created ML environment, run from the cloud GPU project's root:

```bash
.venv-ml/bin/hf auth login
.venv-ml/bin/hf download lerobot/pi05_base --local-dir ./models/pi05_base
.venv-ml/bin/hf download google/paligemma-3b-pt-224 \
  --include '*token*' '*vocab*' '*.model' 'config.json' 'special_tokens_map.json' \
  --local-dir ./models/paligemma-tokenizer
```

The PaliGemma tokenizer may require approved Hugging Face access. After downloading, set these paths in `configs/autodl.env`:

```bash
export PI05_BASE_MODEL="$PI05_REPO_ROOT/models/pi05_base"
export PI05_TOKENIZER="$PI05_REPO_ROOT/models/paligemma-tokenizer"
```

Training and inference load local assets offline. When reusing an environment, use its `hf` executable. See [model asset preparation](docs/environments.md#model-assets) for details.

</details>

<a id="collection"></a>

## 🎮 Collect demonstrations

On the robot PC, run from the repository root. The examples below consistently use the batch directory `avocado_batch01`:

```bash
./run.sh collect --output-dir ./data/raw/avocado_batch01
```

Confirm that Franka Desk is in **Execution + FCI** mode and that the physical stop switch is available to the operator. Then use the collection GUI in this order:

| Operation | Behavior |
| --- | --- |
| Check previews and task | Verify both camera views, joint plots, and the language instruction |
| Start teleoperation (`启动遥操作`) | Let GELLO control the FR3 and wait for actual state readiness |
| Start recording (`开始录制`) | Create a new `record_NNN` episode |
| Complete the pick, placement, and retreat | Keep the full gripper opening and object release |
| Stop recording (`停止录制`) | Save the episode while teleoperation remains active |
| Stop teleoperation (`停止遥操作`) | End robot control, then close the GUI |

The default instruction is **`Put the avocado in the basket.`**, and each episode defaults to `outcome=success`. Set unsuccessful or interrupted demonstrations to `failure` or `aborted` before stopping the recording. The operator must confirm success from the actual task outcome.

**Stopping recording only stops data capture; teleoperation continues.** To end teleoperation completely, you can also run this in another terminal on the robot PC:

```bash
./run.sh stop-collect
```

Each episode includes both videos, saved images, timestamps, JSON/HDF5 state and action data, and completeness metadata:

```text
data/raw/avocado_batch01/record_001/
├── metadata.json
├── action/
│   ├── action_data_*.json
│   └── action_data_*.h5
└── video/
    ├── camera_1_*.mp4
    ├── camera_2_*.mp4
    ├── camera_*_frames/frame_*.jpg
    └── frame_timestamps.json
```

The cameras capture RGB at **30 FPS**, and demonstrations are saved at **15 FPS**. The gripper uses **Move at 0.04 m/s**. The policy learns total opening width; Move has no grasp-force parameter.

[Teleoperation configuration, recording, and stopping →](docs/collection.md)

<a id="dataset"></a>

## 🎬 Review and convert data

After recording, close the collector, make an independent backup of the raw batch, and review both videos episode by episode. Check grasp retention, release, and retreat. A reviewed `--trim-plan` can remove stationary frames at the beginning or end; the converter trims states, actions, and images together and recomputes statistics. See the [manual trimming plan](docs/data.md#manual-trim-plan).

### 1. Validate raw recordings

```bash
./run.sh validate \
  --input-dir ./data/raw/avocado_batch01 \
  --output-dir ./data/lerobot/fr3_avocado_v30 \
  --report ./runs/avocado_batch01_validation.json
```

By default, validation selects **complete recordings labeled successful** and checks states, actions, timestamps, camera order, and video frame counts. The report should contain a nonempty `selected` list and a positive `total_frames`. Inspect the reasons in `skipped` and `errors` before converting.

### 2. Export LeRobot Dataset v3.0

```bash
./run.sh convert \
  --input-dir ./data/raw/avocado_batch01 \
  --repo-id local/fr3_avocado_pi05 \
  --output-dir ./data/lerobot/fr3_avocado_v30
```

Both validation and conversion require a dataset output directory that does not yet exist. Conversion creates a new directory and uses all eligible episodes for training, with no validation split by default. Trimming requires an explicitly supplied reviewed plan; default conversion validates and filters records.

`--input-dir` must directly contain the `record_*` directories. When collecting into `data/raw/avocado_batch01`, use that same batch path for conversion.

```text
data/lerobot/fr3_avocado_v30/
├── data/chunk-000/file-*.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   ├── episodes/
│   └── conversion_manifest.json
└── videos/
    ├── observation.images.camera_1/
    └── observation.images.camera_2/
```

| Feature | Dimensions / specification | Meaning |
| --- | --- | --- |
| `observation.images.camera_1` | 640 × 480 RGB | First camera; default serial `233522074047` |
| `observation.images.camera_2` | 640 × 480 RGB | Second camera; default serial `405622074065` |
| `observation.state` | 8 | Seven measured joint angles (rad) + measured total gripper opening (m) |
| `action` | 8 | Seven absolute joint targets (rad) + the last accepted gripper target width (m) |
| `task_index` | 1 | Language instruction in the task table |
| Recording / training rate | 15 FPS | Frame-aligned states, actions, and both images |

The total gripper opening ranges from **0 to 0.08 m**. When holding an object, the measured and requested widths can differ; they are stored in state and action, respectively. Joint angles and widths are stored in physical units, then normalized using dataset statistics by the training processors.

Upload the **complete** `fr3_avocado_v30/` directory to `data/lerobot/` on the cloud GPU server. Keep `meta/`, `data/`, `videos/`, and conversion provenance together. An already cleaned v3.0 dataset can be used directly, with no second conversion or `clean` suffix required.

[Data contract, backup, trimming, and transfer →](docs/data.md)

<a id="training"></a>

## 🧠 Train and resume π0.5

Run the following commands from the repository root on the **cloud GPU server**, after configuring the local base model and tokenizer:

```bash
./run.sh check-train --dataset ./data/lerobot/fr3_avocado_v30
./run.sh train --dataset ./data/lerobot/fr3_avocado_v30 --dry-run
```

`check-train` checks data, video samples, statistics, action chunks, weights, and the ML runtime. `--dry-run` prints the training configuration. For a new setup, follow the [training guide](docs/training.md) to run the two-step training and resume checks first.

If using W&B, run `wandb login` in the selected ML environment. Start training with:

```bash
./run.sh train \
  --dataset ./data/lerobot/fr3_avocado_v30 \
  --offline \
  --gradient-checkpointing false \
  --batch-size 16 \
  --num-workers 8 \
  --steps 20000 \
  --save-freq 2000 \
  --log-freq 10 \
  --wandb true \
  --out ./outputs/fr3_avocado_pi05
```

The base model comes from `PI05_BASE_MODEL` in `configs/autodl.env`; override it with `--policy-path /absolute/path/to/base-model` if needed. `--data-root` and `--dataset` are aliases, as are `--output-dir` and `--out`.

| Setting | Training configuration |
| --- | --- |
| Trainer | Official LeRobot 0.6.1 trainer / Accelerate |
| Total updates / batch size | 20,000 / 16 |
| Workers / seed | 8 / 1000 |
| Checkpoint / log interval | 2,000 / 10 steps |
| Precision | CUDA + bfloat16 |
| Gradient checkpointing | Disabled |
| Vision encoder | Frozen; `train_expert_only=false` |
| Action chunk | 50 steps |
| Optimizer and schedule | AdamW, lr 2.5e-5, official π0.5 cosine preset |
| Dataset split | All eligible episodes; no validation split |

The example explicitly enables W&B; it is disabled by default. Use `--wandb false` when logging is unnecessary. A fresh training run requires a new output directory.

### Resume training

```bash
./run.sh resume \
  --resume-from ./outputs/fr3_avocado_pi05/checkpoints/last
```

This restores the model, processors, optimizer, scheduler, RNG, and official step count while retaining the saved training plan. You can also select a numbered checkpoint such as `checkpoints/008000`. Resuming an 8k checkpoint from a 20k plan performs the remaining 12k updates; keep the original total step count.

```text
outputs/fr3_avocado_pi05/checkpoints/008000/
├── pretrained_model/     # Model, config, pre/postprocessors, and statistics
└── training_state/       # Optimizer, scheduler, RNG, and training step
```

**Deployment needs the complete `pretrained_model/`; resuming training needs the entire checkpoint.** Base `pi05_base` initialization and native checkpoint loading use separate adapters. The official trainer owns the training loop, sampling, saving, and recovery.

[Training parameters, smoke checks, and checkpoint relocation →](docs/training.md)

<a id="artifacts"></a>

## 🤗 Models and datasets

| Artifact | Hugging Face link |
| --- | --- |
| LeRobot v3.0 robot demonstration dataset | [Dataset](https://huggingface.co/datasets/Metroooooooo/fr3-gello-avocado-v30) |
| π0.5 fine-tuned 12k checkpoint | [Model](https://huggingface.co/Metroooooooo/pi05-fr3-gello-avocado-12k) |
| Raw teleoperation recordings | [Raw Dataset](https://huggingface.co/datasets/Metroooooooo/fr3-gello-avocado-raw) |

<a id="deployment"></a>

## 🚀 Deploy π0.5

The **cloud server runs inference and the robot PC executes actions**. The cloud GPU loads the fine-tuned model, while the robot PC sends both camera images and measured state, receives absolute joint targets and gripper width, and executes them through local ROS2. GELLO supplies collection demonstrations; deployment uses the cameras and FR3 control stack.

### 1. Cloud GPU: start the inference service

Check and serve the **12k checkpoint**:

```bash
./run.sh serve \
  --checkpoint ./outputs/fr3_avocado_pi05/checkpoints/012000/pretrained_model \
  --check-only

./run.sh serve \
  --checkpoint ./outputs/fr3_avocado_pi05/checkpoints/012000/pretrained_model \
  --host 127.0.0.1 --port 8000
```

The service loads the checkpoint's saved processors and statistics. To use a different training step, change the checkpoint path. If training and inference share a GPU, check available memory first.

### 2. Robot PC: configure SSH

```bash
cp -n configs/autodl.env.example configs/autodl.env
```

Edit `configs/autodl.env` on the **robot PC** with the cloud GPU connection details:

```bash
export PI05_SSH_HOST="YOUR_CLOUD_GPU_HOST"
export PI05_SSH_PORT="YOUR_SSH_PORT"
export PI05_SSH_USER="root"
export PI05_LOCAL_PORT="18000"
export PI05_REMOTE_PORT="8000"
```

Complete SSH host-key verification and key-based login manually, then test noninteractive access:

```bash
source configs/autodl.env
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -p "$PI05_SSH_PORT" "$PI05_SSH_USER@$PI05_SSH_HOST" true
```

Use the SSH login port supplied by your cloud GPU provider. `18000` and `8000` are the local forwarded port and cloud inference port, respectively. The deployment supervisor manages the SSH tunnel by default.

### 3. Robot PC: prepare and launch

End teleoperation collection and any other control session, then run:

```bash
./run.sh prepare
./run.sh deploy --network-check
./run.sh deploy --check
./run.sh deploy
```

`prepare` builds and tests the separate deployment overlay. `--network-check` checks SSH and the model service; `--check` checks local configuration and dependencies. Deployment settings are in `deploy/configs/deploy.yaml`. If the FR3 IP changes, also update `robot.ip` there or configure your own settings file as described in the guide.

Use the deployment GUI in this order:

| Button | Behavior |
| --- | --- |
| Connect and preview (`连接并预览`) | Show both camera views and actual state while the model remains disabled |
| Predict only (`仅预测（不运动）`) | Inspect a prediction for the current observation |
| Enable model execution (`启用模型执行`) | Confirm on-site conditions, enable model control, and start recording |
| Stop model execution (`停止模型执行`) | Stop execution, save recordings, and retain the window and previews |
| Open gripper, 80 mm (`打开夹爪（80 mm）`) | Open the gripper separately after model execution is disabled and stopping has completed |

Before motion, confirm Desk Execution / FCI, the scene and robot starting pose, and the availability of the physical stop switch. Collection and deployment share a control lease; only one session can own the robot at a time.

The current implementation uses **serial chunked inference**: predict 50 actions, execute an aligned prefix of at most 6 steps, and infer again from new observations. The robot PC retains interpolation, speed limits, state freshness checks, and tracking checks. RTC is not implemented.

To fully end deployment, run in another terminal:

```bash
./run.sh stop-deploy
```

### 🎥 Save execution videos automatically

Each time model execution is enabled, both cameras start a new recording. Stopping execution, a protective stop, or closing the GUI ends and saves it. Preview, prediction-only, and standalone gripper opening do not record.

```text
deploy/recordings/<time>_<session>_<unique-id>/
├── camera_1.mp4
├── camera_2.mp4
├── frames.jsonl
├── metadata.json
└── encoder logs
```

Recordings include both real RGB videos, frame timestamps, frame and drop counts, and the stop reason. Set `PI05_RECORDING_DIR` in `configs/local.env` to change the destination. These recordings support deployment review and are stored separately from training demonstrations.

[SSH, GUI, gripper handoff, recording, and troubleshooting →](docs/deployment.md)

<a id="verification"></a>

## 📈 Validation scope

The repository includes records of:

- Fresh robot/converter environment installation and dependency consistency checks.
- Synthetic two-camera LeRobot v3.0 export, video decoding, and training reader readback.
- CPU regressions for collection sessions, native checkpoint adaptation, deployment state, and two-camera recording.
- Dedicated ROS image build, compilation of libfranka, 6 Franka ROS packages, and 4 custom packages, plus dynamic linking checks.

The new image still requires on-site hardware acceptance. Complete GPU training, inference, and physical task performance require validation in the corresponding environment. See the [validation record](docs/verification.md) for passed checks, skips, and their scope.

```bash
./run.sh test
```

This entry point runs hardware-free regressions. Missing optional dependencies are reported as skipped tests. See `./run.sh help` for all commands.

<a id="documentation"></a>

## 📖 Documentation

| Guide | Contents |
| --- | --- |
| [Environment setup](docs/environments.md) | Ubuntu, real-time kernel, FR3 networking, Docker / VS Code, uv, cameras, and model assets |
| [Teleoperation and collection](docs/collection.md) | GELLO calibration, GUI, labels, stopping teleoperation, and raw recordings |
| [Data and conversion](docs/data.md) | 8D contract, backup, reviewed trimming, LeRobot v3.0, and provenance |
| [Training and resume](docs/training.md) | Official trainer, 20k / batch 16, processors, checkpoints, and recovery |
| [Two-machine deployment](docs/deployment.md) | Cloud GPU service, SSH, preview, execution, gripper opening, recording, and logs |
| [Dependencies](requirements/README.md) | Three Python profiles, dependency locks, and environment reuse |
| [ROS source preparation](docker/README.md) | Pinned commits, SHA256, source cache, and download retries |
| [Troubleshooting](docs/troubleshooting.md) | FCI, cameras, gripper, data, training, and deployment issues |

<a id="acknowledgements"></a>

## 🙏 Acknowledgements and licensing

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot): π0.5 integration, datasets, processors, and official training infrastructure.
- [GELLO](https://github.com/wuphilipp/gello_software): teleoperation hardware and software.
- [Franka Robotics](https://github.com/frankarobotics): libfranka and franka_ros2.
- [RealSense SDK](https://github.com/realsenseai/librealsense): RGB camera interfaces.
- [data-collection](https://github.com/renshaojie233/data-collection): the upstream collection GUI and synchronized recording foundation.
- [Pi05-Mujoco](https://github.com/metro8116/Pi05-Mujoco): the companion simulation workflow and the organization reference for this README.

Each component retains its original copyright, LICENSE, and NOTICE. See [licensing and provenance](LICENSE.md) for sources and applicable terms.
