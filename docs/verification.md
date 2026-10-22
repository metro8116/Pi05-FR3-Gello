# Verification Scope and Results

English · [简体中文](verification_zh.md)

Date: 2026-10-22. This repository brings the required source from four currently usable local projects into one repository, then organizes entry points, paths, dependencies, documentation, and tests. It has not executed real hardware motion. Successful historical trials of the original projects do not constitute installation acceptance for the new image.

The initial migration checks used existing Python environments and inspected dependencies without installing, upgrading, or changing those environments. Later installation checks in new temporary environments and the ROS image are described in the updates below. CPU tests use synthetic data, temporary directories, and fake bridges, cameras, and device backends. They do not open physical cameras, serial devices, or the arm.

The following areas have been verified:

- New entry-point help, read-only doctor checks, argument forwarding, relocation to a repository path containing spaces, and protection of existing environments.
- Shell syntax, Python source syntax, Compose configuration parsing, and tutorial links.
- Headless collection data contracts; filtering by session owner, PID, and process start time; retention of session records when stopping fails; and no process scan when no session exists.
- Converter regressions covering 18 filtering, corruption, and no-write cases; export of a genuinely encoded, synthetic two-camera, 12-frame schema-2 record to v3.0; and readback and decoding through LeRobot. The original input digests remained unchanged.
- Official training entry points and arguments, processors, the distinction between base and native loading, strict weight keys/shapes/aliases, and native save/resume with a small model. The initial migration checks included LeRobot 0.4.4, so the corresponding 0.6.1-specific remapper tests were skipped in those historical results. The converter default has since been standardized on LeRobot 0.6.1 / Python 3.12; fresh installation and export/readback results are recorded below.
- CPU regressions for the existing deployment state gates, gripper handoff, watchdogs, supervised cleanup, retained fault windows, and standalone bundles; plus object-identity checks for the shared control lock and shared native loader.

Tests with missing ML/camera dependencies, or without a supplied real dataset, are reported as skipped. Initial migration and added recording results are in `docs/verification_results.json`; subsequent focused checks are described below. Read the skip reasons rather than replacing them with a blanket claim that everything passed.

No fresh environment installation was performed during the initial migration. Dependencies were subsequently corrected, and robot and updated converter profiles were installed in separate temporary environments. The ROS image was then rebuilt, including compilation of libfranka and the Franka ROS underlay, as described below. Full GPU PI05 numerical training/resume/inference, a deployment SSH tunnel, and real hardware or camera tests still have not been performed. Dependency work included a read-only comparison of versions and package-file hashes on the existing cloud GPU server, without installing packages, changing them, or controlling the training process. Deployment recording integrity was checked with CPU fake writers and actual software MP4 encoding/decoding. Real USB throughput, available disk capacity, and recording load still need observation during the next on-site run.

For subsequent acceptance, follow the tutorials through `doctor`, `collect --check`, ROS build/prepare, data preflight, a two-step training/resume check, robot-free network checks, and preview. Only then should the on-site operator enable the model. Do not perform this acceptance by installing or rebuilding inside an old environment that is actively training.

There is one exception to leaving the original projects unchanged: the user subsequently requested **automatic two-camera deployment recording** in the existing Pi05-deploy project as well. Only the corresponding feature, configuration, tests, and documentation in that deployment directory are updated. The original GELLO, collection, and training projects, existing data, and checkpoints remain unchanged.

## Converter Environment Update to 0.6.1

The NumPy dependency conflict has been fixed. Conversion now defaults to Python 3.12 / LeRobot 0.6.1 / CPU Torch 2.11, while training continues to use the corresponding CUDA build. All three profiles have complete lock files. The ML Torch-index issue that shadowed PyPI was resolved with direct links to official wheels and their SHA256 hashes.

A fresh temporary venv was used to install the robot profile's 7 dependencies, the converter profile's 65 dependencies, and this project. All `uv pip check` checks passed. The robot profile passed the hardware-free `collect --check` import check. All 4 updated converter tests passed, covering 18 raw-data validation cases, export of a 12-frame synthetic two-camera record to v3.0, decoding/readback, training-data preflight, and padding the final 50-step action chunk. These tests used complete avocado-task/Move metadata and did not read or modify actual collection records.

At the user's request, the newly created `.venv-robot` and `.venv-converter` directories in the new repository were deleted, and the temporary verification environments were also cleaned up. The user will create the production environments from scratch. Detailed results are in [converter_environment_061_verification.json](converter_environment_061_verification.json).

## ROS Image Download Fix and Actual Build

To address GitHub clone failures with `GnuTLS recv error (-110)`, host-side source preparation, four commit-pinned SHA256 locks, and a cache were added, including libfranka's common submodule. The image only verifies and unpacks these sources offline. An explicit hardware dependency list replaces rosdep's online GitHub-index download. Completed base apt/Python build layers were reused.

On 2026-10-22, `./run.sh ros-image` was actually run and successfully produced `pi05-fr3-gello:humble`, with image ID `sha256:fbb3d436f260c0367fefa84fd26d1bb48420c970905c0cead620e58c71b2bf57`. libfranka 0.15.0 and 6 Franka ROS packages compiled successfully. The source-preparation tool passed 17 tests under system Python 3.8, all four actual source archives passed offline verification, and Shell syntax and Compose configuration parsing passed.

A temporary container with no network, device mounts, or privileges was then used to check paths for 8 ROS packages, the FR3 URDF, and dynamic linking for 7 binaries/shared libraries. All checks passed. The new repository's source was subsequently mounted read-only, and 4 custom packages were actually compiled with `BUILD_TESTING` enabled: the GELLO publisher, FR3 arm controller, gripper manager, and pi05 gripper driver. Compilation and package queries passed. Their runtime tests and all ROS control nodes were not run during this check. The temporary container was removed, and no persistent host overlay was created. Production use still requires `ros-up` and `ros-build` as described in the tutorials, plus `prepare` for deployment.

No real arm, camera, or serial device was started or connected. No original project, existing Python environment, data, weights, or cloud GPU server training was modified. Full results are in [ros_image_verification_20261022.json](ros_image_verification_20261022.json).
