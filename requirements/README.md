# Separate dependency profiles

English · [简体中文](README_zh.md)

The same codebase uses three separate Python environments. Do not combine these requirements or install/upgrade packages in an old environment that is currently training. `robot.txt`, `converter.txt`, and `ml.txt` pin key dependencies; each corresponding `*.lock.txt` pins the complete resolved dependency set. `setup` prefers the lock file and then checks dependency consistency.

| Files | Python | Purpose and verification basis |
| --- | --- | --- |
| `robot.txt` / `robot.lock.txt` | 3.10 | Local collection, dual RealSense preview, and the remote-deployment client. Versions come from the working `take_data` environment. ROS2/C++ drivers run in a separate container. |
| `converter.txt` / `converter.lock.txt` | 3.12, Linux x86_64, glibc >=2.28 | Convert raw schema-2 recordings to Dataset v3.0. Uses the same LeRobot **0.6.1** release as training, with CPU PyTorch and PyAV for conversion. |
| `ml.txt` / `ml.lock.txt` | 3.12 | Training and inference on the cloud GPU server. Reuses the existing LeRobot **0.6.1**, Torch **2.11.0+cu128**, and Transformers **5.5.4** combination. |

Dataset **v3.0** is the on-disk data format; `lerobot==0.6.1` is the Python package version. Conversion and training now use the same LeRobot release, but conversion needs only the CPU. Collection, conversion, and training do not need to share one Python environment.

For example, create a new isolated environment from the project root:

```bash
uv venv --python 3.12 .venv-converter
uv pip install --python .venv-converter/bin/python -r requirements/converter.lock.txt
uv pip check --python .venv-converter/bin/python
CUDA_VISIBLE_DEVICES='' .venv-converter/bin/python -m unittest discover -s tests -p test_converter_cli.py -v
```

These commands create and install into a new isolated environment only. The robot and ML profiles must likewise use their own new environments. Official CUDA wheel URLs and SHA256 hashes for ML are recorded in `ml.txt`; the host NVIDIA driver must support CUDA 12.8. Tkinter, USB/camera permissions, Docker, ROS2, and the real-time kernel are covered by the main host/container setup guide and cannot be replaced by these pip requirements files.

### Converter dependency choices

LeRobot 0.6.1 requires Python >=3.12 and NumPy >=2,<2.3. The converter profile pins `numpy==2.2.6`, `opencv-python-headless==4.13.0.92`, and `h5py==3.15.1`. Do not install this converter profile into a Python 3.10 environment or use `--no-deps` to bypass conflicts.

Conversion needs only the CPU. `torch==2.11.0+cpu` and the matching `torchvision==0.26.0+cpu` use official Python 3.12 / Linux x86_64 wheel URLs with SHA256 hashes. They require glibc >=2.28; the current Ubuntu 20.04 host's glibc 2.31 satisfies this requirement. Torch/Torchvision use the same software releases as the ML profile, with CPU builds for conversion. There is no additional CUDA wheel index, and no NVIDIA CUDA or Triton packages are introduced.

Data writing and reading use explicit PyAV dependencies: `datasets==4.8.5`, `pandas==2.3.3`, `pyarrow==26.0.0`, `av==15.1.0`, `jsonlines==4.0.0`, `imageio==2.38.1`, and `imageio-ffmpeg==0.6.0`. The `[dataset]` / `[training]` extras that would introduce TorchCodec are not enabled. PI05/Transformers extras are also omitted, and no model is loaded. Conversion and readback explicitly select PyAV; other tools that retain LeRobot's default TorchCodec backend must be configured to use PyAV separately.

On 2026-10-22, uv 0.10.12 resolved all 65 dependencies for Python 3.12 and the host's glibc 2.31 platform. The profile also passed `--only-binary :all:` resolution, confirming a compatible wheel for every dependency. Hardware and visualization are optional dependencies in 0.6.1, so this converter profile does not introduce `pynput`, `evdev`, or `rerun-sdk`. All 65 packages were installed in a fresh temporary local Python 3.12 CPU environment, and `uv pip check` passed. A synthetic dual-camera recording was successfully exported to v3.0, decoded, and read back. You can still run the checks and synthetic tests above after your own installation.

After changing key dependencies, regenerate and review the lock, then install it in a new environment and check it. Do not merely edit a single version inside the lock:

```bash
uv pip compile requirements/converter.txt \
  --python-version 3.12 --python-platform x86_64-manylinux_2_31 \
  --only-binary :all: --no-strip-extras --output-file requirements/converter.lock.txt
```

## Conversion workflow

The default input is the repository's `data/raw` directory, and the default output is a new `data/lerobot/fr3_avocado_v30` directory. An existing output is not overwritten. Only complete, successful schema-2 episodes are selected by default; all selected episodes are used for training, with no validation split.

```bash
.venv-converter/bin/python tools/convert_to_lerobot_v30.py \
  --input-dir /path/to/raw --dry-run --report /path/to/validation-report.json

.venv-converter/bin/python tools/convert_to_lerobot_v30.py \
  --input-dir /path/to/raw --output-dir /path/to/new-dataset
```

A report must contain `status=validated`, a nonempty `selected` list, and a positive `total_frames` before there is data to export. `--dry-run` does not import LeRobot, create an output directory, or connect to cameras or the robot. The optional `--trim-plan` accepts a reviewed trimming plan; the tool does not infer trims or delete original data itself.

## LeRobot 0.6.1 and Transformers compatibility

A read-only audit of the existing cloud GPU server environment on 2026-10-22 found Python 3.12.3 and key package versions matching `ml.txt`. All 2,271 Transformers `.py` files and all 490 LeRobot `.py` files matched the hashes in their installed wheels' `RECORD` files. No old-style `transformers_replace` patch or manual package-source modification was found. This profile therefore **does not add a Transformers replacement script**. Do not apply a 4.x Transformers replacement patch from an older tutorial to this 5.5.4 environment.

The repository's own weight-loading adaptation is still required: initial `pi05_base` loading permits base-weight key migration, while official training-checkpoint recovery strictly loads native keys and preserves independent `lm_head` and `embed_tokens` weights. This code belongs to the project's `src/fr3_pi05` package and does not modify the installed LeRobot/Transformers packages.

The converter has checked the 0.6.1 signatures of `LeRobotDataset.create`, `RGBEncoderConfig`, `add_frame`, `save_episode`, and `finalize`; its existing dynamic argument adaptation matches these interfaces. The converter profile separately supplies h5py and the PyAV data stack. Actual synthetic export and video readback tests used a newly created local temporary CPU environment, avoiding installation of conversion dependencies into the cloud GPU server environment that is currently training.

ML uses explicit PyAV data dependencies without the `[training]` extra that would additionally install TorchCodec. The tokenizer and model are read offline from existing local directories/caches; project checks do not automatically download weights. After creating a new ML environment, run the project environment checks and small CPU regressions first. The operator can then explicitly run training or a server.

Deployment's automatic video recording also uses the FFmpeg executable supplied by `imageio-ffmpeg==0.6.0`, which is included in the robot profile. When reusing an existing robot environment without that package, set `PI05_FFMPEG` to an already available FFmpeg executable. Recording does not require modifying the old training environment. This added dependency is explicitly installed only for new environments; no packages were installed into the old robot environment during this work.
