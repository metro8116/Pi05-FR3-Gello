# Environment Setup from Scratch and Reuse of Existing Environments

English · [简体中文](environments_zh.md)

This guide targets Linux x86_64. The robot computer and cloud GPU server use the same repository, with environments separated by role. New installation commands operate only on this repository's `.venv-*` environments, image, and ROS workspace; they do not modify the four original projects. If the existing projects already work on this machine, prefer explicitly reusing their environments before reinstalling drivers or kernels.

## 1. System prerequisites on the robot computer

Ubuntu 22.04 LTS is recommended for a new machine. A working Ubuntu 20.04 host can remain in use, with ROS Humble running inside an Ubuntu 22.04 container. Containers share the host kernel: **installing Docker does not replace the host real-time kernel and scheduling configuration required for Franka real-time control**. Configure the real-time kernel according to the [official Franka installation guide](https://frankarobotics.github.io/docs/installation_linux.html) and the Robot System/libfranka versions supported by the device. Do not apply mismatched kernel packages or firmware to a working machine.

Inspect the current system first:

```bash
uname -a
cat /sys/kernel/realtime 2>/dev/null || true
ip -br addr
ls -l /dev/serial/by-id/
```

Install basic host tools as an operator with administrator privileges:

```bash
sudo apt update
sudo apt install -y git curl build-essential pkg-config ffmpeg \
  libgl1 libglib2.0-0 libusb-1.0-0 udev python3-tk openssh-client
```

Install [Docker Engine and the Compose plugin](https://docs.docker.com/engine/install/ubuntu/), then check `docker info` and `docker compose version`. Configure access for an ordinary user as described in Docker's official documentation, then log in again. Docker access is effectively host administrator access; do not expose the Docker socket to the public internet. This project explicitly uses `unix:///var/run/docker.sock` to avoid a different Docker context selecting an engine without access to the devices.

## 2. FR3 networking and physical operating mode

The current configuration uses FR3 address `172.16.0.2` and `172.16.0.100/24` on the robot computer's dedicated wired interface. In Ubuntu network settings, assign static IPv4 to the interface **actually connected to the robot**. This dedicated connection normally has no default gateway; internet traffic continues through another interface. Do not apply the example address to the wrong interface.

```bash
ip route get 172.16.0.2
ping -c 3 172.16.0.2
```

Open the device's Desk address in a browser, typically `https://172.16.0.2`. Follow the normal procedure for its current software version to unlock joints, release the physical stop condition, select **Execution**, and enable **FCI**. Do not bypass robot stop or collision protection. `UserStopped` means the device is still receiving a stop signal; reopening the collection window alone will not clear it.

Set `FRANKA_IP` in `configs/local.env` for collection. For deployment, `robot.ip` in the default `deploy/configs/deploy.yaml` must match it. To use a private copy such as `deploy/configs/local.yaml`, point `PI05_CONFIG` to that file. The robot computer maintains real-time control; the cloud GPU server only receives images/state and returns lower-frequency targets. The server does not need direct access to the FR3 private network.

## 3. GELLO, RealSense, and gripper fingers

Use a stable `/dev/serial/by-id/...` path for GELLO rather than relying on `/dev/ttyUSB0`, which may change after reboot. Check device permissions. If necessary, have an administrator add the current user to `dialout`, then log in again. With multiple serial devices, set this in `configs/local.env`:

```bash
export GELLO_PORT="/dev/serial/by-id/YOUR_GELLO_DEVICE"
```

The repository's `ros2_ws/src/franka_gello_state_publisher/config/my_gello.yaml` is the historical configuration for the current GELLO device, including seven joint directions, offsets, and gripper travel. Do not copy it unchanged to another device. Determine `joint_signs`, `best_offsets`, and `gripper_range_rad` using the GELLO hardware and upstream calibration instructions. Check values and directions first, then perform a small-motion test at the robot. See [GELLO upstream](https://github.com/wuphilipp/gello_software). This project does not automatically calibrate, home the gripper, or enable Dynamixel torque.

Connect the two RealSense cameras to USB 3 ports with sufficient bandwidth. Avoid sharing an undersized hub with several high-bandwidth devices. Follow the [RealSense SDK Linux documentation](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md) to install SDK/udev rules appropriate for the kernel, then reconnect USB and verify permissions. The project uses RGB only and does not depend on depth streams.

The current mapping is camera_1=`233522074047`, camera_2=`405622074065`. Keep camera position, orientation, and cropping as close as possible to collection conditions. This version's deployment `config.py` also fixes the task's two serial numbers and resolution to prevent accidental viewpoint changes. Replacing cameras requires updating collection's `TAKE_DATA_CAMERA_SERIALS`, the converter's `--camera-serials`, deployment's `cameras.serials`, and that explicit validation, followed by a review of the data/training contract. Swapping code order alone is insufficient, and an existing model cannot simply be assumed to work from a new viewpoint.

The default gripper policy is **Move / 0.04 m/s / total width 0–0.08 m**, not 5 N force control. When Move encounters an obstacle, actual contact force is not limited by the model's width output. After changing fingers, materials, or structure, verify the hardware under its load limits at the robot, and retain any hardware fault lock until inspection and repair are complete. This repository does not clear the lock or perform extra closing tests.

## 4. uv and Python environments

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then open a new terminal or load PATH as directed by the installer. The project uses uv to create ordinary isolated venvs; neither conda nor `uv sync` is required. Select environments by machine role:

```bash
cd /path/to/Pi05-FR3-Gello
./run.sh setup robot      # Host GUI / cameras, Python 3.10
./run.sh setup converter  # Data conversion, Python 3.12 / LeRobot 0.6.1
# Run only on the cloud GPU server:
./run.sh setup ml         # Python 3.12 / LeRobot 0.6.1
```

Dependency details and verification evidence are in [requirements/README.md](../requirements/README.md). `setup` automatically selects each profile's complete lock file. Conversion and training use Python 3.12 / LeRobot 0.6.1; robot continues to use Python 3.10. By default, setup refuses to modify an existing `.venv-*` with the same name, and it does not install into the currently activated environment. A network or installation failure leaves the environment in place rather than deleting it automatically. After fixing the cause, continue with `./run.sh setup converter --resume`, substituting the relevant profile name for other roles. `--resume` accepts only a real venv in the corresponding directory within this repository, with the correct interpreter prefix and Python version. It rejects symlinks to other environments. It installs or updates dependencies in that target environment; it is not the command for reusing an old environment.

Tkinter is an OS/Python component. If `import tkinter` fails in a uv-managed Python, install matching Tcl/Tk and Python Tk support, or manually create `.venv-robot` with a Python 3.10 interpreter known to include Tk, then install `requirements/robot.txt`. Installing GUI packages into the ML environment does not resolve the robot computer's environment problem.

### Reuse a working environment directly

No activation or installation is required. After copying the templates, set actual Python paths in `configs/local.env` / `configs/autodl.env`:

```bash
export PI05_ROBOT_PYTHON="/path/to/existing/robot-env/bin/python"
export PI05_CONVERTER_PYTHON="/path/to/existing/converter-env/bin/python"
export PI05_ML_PYTHON="/path/to/existing/ml-env/bin/python"
```

These interpreters run from their original locations while the project source comes from the new repository. `run.sh` does not invoke pip/uv except through explicit `setup`. Reusing an old environment neither migrates it nor changes an active training process. Exit old collection/deployment control sessions normally before using the new program; the shared kernel lock prevents concurrent ownership.

## 5. ROS container and VS Code

```bash
./run.sh ros-image   # Download and compile; first build takes time, starts no robot nodes
./run.sh ros-up      # Start an idle container running sleep only
./run.sh ros-build   # Build the collection overlay
./run.sh prepare     # Build/test the separate deployment overlay
```

The separate container is named `pi05-fr3-gello-ros2-1`. Host `ros2_ws` is mounted at `/workspace/ros2`, and the repository at `/workspace/repo`. The container uses host networking and USB with real-time scheduling capabilities retained. The image contains no data or models and does not depend on the old `gello_software` checkout. `ros-up` itself does not run robot control.

The ROS underlay pins Git commits for libfranka **0.15.0**, franka_ros2 **v2.0.4**, and franka_description **1.0.2**; see `docker/Dockerfile`. The `package.xml` files in v2.0.4 still declare 2.0.2, which does not indicate an incorrect installation. The new image was actually built on 2026-10-22: libfranka and six Franka ROS packages compiled successfully. Package queries and dynamic-link checks passed in a temporary container without devices or network; see the [verification record](verification.md). Hardware has not yet been tested with this new image. Check Franka firmware compatibility against the official matrix.

`ros-image` first downloads and verifies official source archives for four fixed commits, including the common submodule, on the host and caches them in `docker/vendor/`. The container extracts and compiles them, avoiding TLS interruptions during a full GitHub clone. Verified source archives and successful Docker layers are reused. Source downloads and the later Franka ROS apt dependency installation have retries. If the initial base apt stage fails, rerun the unified entry point. The image explicitly lists apt dependencies for real robot control and no longer fetches rosdep's GitHub index. The cache is excluded from Git; see [image source preparation](../docker/README.md).

Install the Dev Containers extension in VS Code, open this repository, and select **Dev Containers: Reopen in Container**. `.devcontainer/devcontainer.json` uses the same Compose service. The GUI runs on the host; the VS Code container is for ROS development, and does not need another Torch installation. Closing VS Code does not mean the robot has stopped; use the program's stop command first. Container-generated build/install directories may be owned by root. Continue building them inside the container rather than manually overwriting the original workspace.

<a id="model-assets"></a>

## 6. New cloud GPU server environments and model assets

Choose an NVIDIA driver/GPU compatible with the current PyTorch CUDA wheel on the cloud GPU server. Use `nvidia-smi` to inspect the driver and free GPU memory. The observed ML profile version is Torch **2.11.0+cu128**. A system CUDA toolkit installation is generally not required to use this wheel, but the host driver must be compatible. Batch size 16 with gradient checkpointing disabled uses substantial memory. Run the explicit smoke test described in the training guide first; the GPU model alone does not guarantee that the workload fits.

Existing working uv environments, base weights, and tokenizer caches can be reused. Do not upgrade an environment while it is training. On a fresh machine, create the ML environment, then prepare the model and tokenizer with network access:

```bash
# In the cloud GPU server's repository; accept applicable gated-model terms on HF and log in first.
.venv-ml/bin/hf auth login
.venv-ml/bin/hf download lerobot/pi05_base --local-dir ./models/pi05_base
.venv-ml/bin/hf download google/paligemma-3b-pt-224 \
  --include '*token*' '*vocab*' '*.model' 'config.json' 'special_tokens_map.json' \
  --local-dir ./models/paligemma-tokenizer
```

Do not commit tokens or `models/`. The base directory must contain `config.json` and a single `model.safetensors`; the current entry point does not support sharded or adapter-only assets. When reusing the HF cache, point to `models--lerobot--pi05_base/snapshots/<hash>`, not the outer cache root. Use the tokenizer and weight-header checks in `./run.sh check-train` to establish which assets are actually usable. If the tokenizer download filters miss a dependency, follow the error to supply the complete tokenizer configuration/vocabulary files; downloading the entire vision-language model's weights solely for its tokenizer is unnecessary.

Set the following in `configs/autodl.env`:

```bash
export PI05_BASE_MODEL="$PI05_REPO_ROOT/models/pi05_base"
export PI05_TOKENIZER="$PI05_REPO_ROOT/models/paligemma-tokenizer"
```

Training and inference read local assets in offline mode by default, without automatic downloads at startup. W&B is optional. Before enabling it, run `wandb login` in the ML environment and keep credentials in the machine's private configuration.

## 7. Post-installation checks

```bash
./run.sh doctor                 # Dependency/path inventory without hardware
./run.sh collect --check        # Robot-computer Python imports; does not open cameras
./run.sh check-train --data-only # Check on the GPU host after the complete dataset is present
./run.sh deploy --network-check # After service startup and SSH setup; does not connect to the robot
```

Missing entries in `doctor` may correspond to profiles installed only on the other machine. It does not automatically fix dependencies, reconnect FCI, ask for passwords, open cameras, or allocate a large model. Use the role-specific checks to establish whether the corresponding environment is ready.

## 8. Automatic deployment recording

The robot profile includes imageio-ffmpeg to supply FFmpeg in a new isolated environment. When reusing an old environment without a system FFmpeg executable, set `PI05_FFMPEG` in private `configs/local.env` to an existing executable; alternatively, create the new robot environment for that role. A recording startup failure prevents model execution from being enabled. Recordings default to `deploy/recordings/`, so ensure sufficient disk space. See the deployment guide for parameters and save states. For the existing Pi05-deploy project, this work only pointed its private configuration to an encoder already present; no package was installed into the old environment.
