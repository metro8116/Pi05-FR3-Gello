# ROS Image Source Preparation and Interrupted Downloads

English · [简体中文](README_zh.md)

`./run.sh ros-image` first runs `prepare_sources.py` on the host to prepare four official source archives pinned to specific commits. `sources.lock.json` records their versions, commits, source URLs, and SHA256 hashes, including the exact common submodule commit used by libfranka. It does not clone full Git history or disable TLS verification.

The cache is stored in this repository's `docker/vendor/` directory. The four official archives total approximately 36.7 MB. Verified archives are reused and included in the Docker build context, but the cache is ignored by Git and is not uploaded when publishing the repository. Initial download failures are retried, and temporary files are never treated as complete source. If an existing cache entry fails verification, it is retained and an error is reported; it is not silently overwritten.

```bash
./run.sh ros-image
```

Docker uses only the verified local sources and verifies them again offline. The versions of libfranka/common, franka_ros2, and franka_description remain pinned. System apt dependencies still require network access. Later Franka ROS dependency stages include retries; if the initial base apt stage fails, rerun the unified entry point. The image explicitly installs the dependencies needed for real hardware control instead of requesting GitHub indexes through rosdep. It builds only franka_gripper, franka_robot_state_broadcaster, franka_description, and their transitive source dependencies, excluding unused Gazebo/MoveIt examples. MoveIt Core/Msgs remain installed because the project's custom controller requires them. The first compilation still takes time: completing the source downloads does not mean all dependency installation has finished.

To prepare sources manually, or reuse matching source archives from another directory:

```bash
python3 docker/prepare_sources.py
python3 docker/prepare_sources.py --offline
python3 docker/prepare_sources.py --cache-dir /absolute/path/to/source-cache
```

Replace the placeholder cache path with your chosen directory. `--cache-dir` may be repeated. It does not scan old projects or modify an imported directory. The lock file also accepts three audited historical archives containing complete Git source: their tracked files are unmodified and their HEAD commits match the pinned commits. The common submodule is still supplied by the separately locked fourth archive. Archives from other sources, or with mismatched hashes, are not used.

After a network interruption, rerun `./run.sh ros-image`. Completed Docker layers are retained; there is no need to rebuild Python environments or run Docker prune. If you invoke compose build directly, run the source-preparation tool first. The unified entry point handles this automatically.
