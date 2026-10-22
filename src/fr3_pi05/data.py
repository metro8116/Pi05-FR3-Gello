"""Read-only FR3 Move dataset contract and all-episode training statistics.

Importing this module requires only the standard library. Parquet, LeRobot,
NumPy, and Torch are imported inside the operations that need them. No dataset
metadata is rewritten, and the converter's optional 45/5 split is not used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


TASK = "Put the avocado in the basket."
STATE_NAMES = [f"fr3_joint{i}" for i in range(1, 8)] + ["gripper_width_m"]
ACTION_NAMES = [f"fr3_joint{i}" for i in range(1, 8)] + ["gripper_target_width_m"]
UNITS = ["rad"] * 7 + ["m"]
CAMERA_KEYS = ["observation.images.camera_1", "observation.images.camera_2"]
ACTION_SEMANTICS = "absolute_FR3_joint_position_targets_and_last_accepted_gripper_width"
CAPTURE_CONTRACT = "fr3_joint_position_v1"
MOVE_PROFILE = {"schema_version": 1, "control_policy": "move_width_v1", "policy_config": None}


class DatasetContractError(ValueError):
    """The local dataset does not satisfy the trained robot interface."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetContractError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"Cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _local_file(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), "Missing relative dataset file path")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), f"Dataset path escapes its root: {relative}")
    _require(path.is_file() and path.stat().st_size > 0, f"Missing or empty dataset file: {path}")
    return path


def _parquet_module():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Dataset preflight needs pyarrow in the selected Python environment") from exc
    return pq


def inspect_contract(dataset_root: str | Path) -> dict[str, Any]:
    """Validate JSON provenance, Parquet rows, task, local videos, and boundaries.

    This checks every scalar row without decoding videos or loading weights.
    Historical absolute source paths in the manifest are provenance only; all
    training inputs must be present underneath ``dataset_root``.
    """
    root = Path(dataset_root).expanduser().resolve()
    _require(root.is_dir(), f"Dataset root does not exist: {root}")
    info = _read_json(root / "meta/info.json")
    manifest_path = root / "meta/conversion_manifest.json"
    if not manifest_path.is_file():
        manifest_path = root / "meta/source_manifest.json"
    manifest = _read_json(manifest_path)
    _require(info.get("codebase_version") == "v3.0", "Expected LeRobot codebase_version v3.0")
    _require(manifest.get("dataset_format") == "v3.0", "Manifest must identify dataset_format v3.0")
    _require(info.get("robot_type") == "franka_fr3", "Expected robot_type franka_fr3")
    _require(info.get("fps") == 15 and manifest.get("fps") == 15, "Expected 15 FPS")
    n_episodes, n_frames = info.get("total_episodes"), info.get("total_frames")
    _require(type(n_episodes) is int and n_episodes > 0, "total_episodes must be positive")
    _require(type(n_frames) is int and n_frames >= n_episodes, "Invalid total_frames")
    _require(info.get("total_tasks") == 1, "This training task requires exactly one task")
    features = info.get("features", {})
    for key, names in (("observation.state", STATE_NAMES), ("action", ACTION_NAMES)):
        feature = features.get(key, {})
        _require(feature.get("dtype") == "float32", f"{key} must be float32")
        _require(feature.get("shape") == [8] and feature.get("names") == names,
                 f"{key} must be the ordered 8D FR3 joint/width vector")
    image_features = [key for key in features if key.startswith("observation.images.")]
    _require(set(image_features) == set(CAMERA_KEYS), f"Expected exactly these cameras: {CAMERA_KEYS}")
    for key in CAMERA_KEYS:
        feature = features[key]
        _require(feature.get("dtype") == "video" and feature.get("shape") == [480, 640, 3],
                 f"{key} must be a 480x640 RGB video")
        video_info = feature.get("info", {})
        _require(video_info.get("video.fps") == 15 and video_info.get("video.channels") == 3,
                 f"{key} must contain 15 FPS, 3-channel video")
    for key, expected in (("capture_contract", CAPTURE_CONTRACT),
                          ("observation_state_features", STATE_NAMES),
                          ("action_features", ACTION_NAMES), ("feature_units", UNITS),
                          ("action_semantics", ACTION_SEMANTICS),
                          ("image_shape_hwc", [480, 640, 3])):
        _require(manifest.get(key) == expected, f"Manifest {key} does not match FR3 contract")
    _require(manifest.get("gripper_control_profiles") == [MOVE_PROFILE],
             "Only the Move width profile is supported; do not silently mix Grasp data")
    mapping = manifest.get("camera_mapping", [])
    _require(isinstance(mapping, list) and len(mapping) == 2, "Missing camera mapping")
    camera_serials = {}
    for i, key in enumerate(CAMERA_KEYS, 1):
        matches = [item for item in mapping if item.get("feature") == key and item.get("camera_index") == i]
        _require(len(matches) == 1 and bool(matches[0].get("serial")), f"Invalid mapping for {key}")
        camera_serials[i] = matches[0]["serial"]
    sources = manifest.get("sources")
    _require(isinstance(sources, list) and len(sources) == n_episodes,
             "Manifest sources do not match total_episodes")
    source_lengths = []
    for i, source in enumerate(sources):
        _require(source.get("episode_index") == i, "Manifest episode indices must be contiguous from zero")
        _require(source.get("task") == TASK and source.get("outcome") == "success",
                 f"Episode {i} is not a successful avocado task")
        _require(source.get("gripper_control_profile") == MOVE_PROFILE, f"Episode {i} is not Move data")
        metadata = source.get("metadata", {})
        for key, expected in (("schema_version", 2), ("capture_contract", CAPTURE_CONTRACT),
                              ("observation_state_features", STATE_NAMES), ("action_features", ACTION_NAMES),
                              ("observation_state_units", UNITS), ("action_units", UNITS),
                              ("action_semantics", ACTION_SEMANTICS), ("task", TASK),
                              ("outcome", "success"), ("status", "complete"),
                              ("gripper_control_profile", MOVE_PROFILE)):
            _require(metadata.get(key) == expected, f"Episode {i} metadata {key} does not match")
        _require(metadata.get("errors") == [], f"Episode {i} has capture errors")
        count = source.get("frames")
        _require(type(count) is int and count > 0 and metadata.get("frame_count") == count
                 and metadata.get("action_count") == count, f"Episode {i} frame counts disagree")
        source_lengths.append(count)
        cameras = metadata.get("cameras", [])
        _require(len(cameras) == 2, f"Episode {i} is missing a camera")
        for number, serial in camera_serials.items():
            _require(sum(c.get("camera_index") == number and c.get("serial") == serial for c in cameras) == 1,
                     f"Episode {i} camera {number} does not match mapping")
        _local_file(root, source.get("source_timestamps_file", ""))
    _require(sum(source_lengths) == n_frames, "Manifest total frame count disagrees with info.json")

    # JSON checks above work without importing the ML stack. Parquet is lazy.
    pq = _parquet_module()
    tasks_path = _local_file(root, "meta/tasks.parquet")
    task_rows = pq.read_table(tasks_path).to_pylist()
    _require(len(task_rows) == 1 and task_rows[0].get("task_index") == 0,
             "tasks.parquet must contain a single task_index=0")
    task_row = task_rows[0]
    task_text = task_row.get("task", task_row.get("__index_level_0__"))
    _require(task_text == TASK, f"Unexpected task in tasks.parquet: {task_text!r}")
    episode_paths = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    _require(bool(episode_paths), "Missing meta/episodes Parquet files")
    episode_rows = [row for path in episode_paths for row in pq.read_table(path).to_pylist()]
    episode_rows.sort(key=lambda row: row["episode_index"])
    _require(len(episode_rows) == n_episodes, "Episode Parquet count disagrees with info.json")
    expected_index = 0
    data_files = set()
    for i, episode in enumerate(episode_rows):
        _require(episode["episode_index"] == i and episode["tasks"] == [TASK],
                 f"Invalid episode metadata at episode {i}")
        length = source_lengths[i]
        _require(episode["length"] == length and episode["dataset_from_index"] == expected_index
                 and episode["dataset_to_index"] == expected_index + length,
                 f"Episode {i} has inconsistent boundaries")
        expected_index += length
        relative = info["data_path"].format(chunk_index=episode["data/chunk_index"],
                                             file_index=episode["data/file_index"])
        data_files.add(_local_file(root, relative))
        for key in CAMERA_KEYS:
            relative = info["video_path"].format(video_key=key, chunk_index=episode[f"videos/{key}/chunk_index"],
                                                  file_index=episode[f"videos/{key}/file_index"])
            _local_file(root, relative)
            start, end = episode[f"videos/{key}/from_timestamp"], episode[f"videos/{key}/to_timestamp"]
            _require(math.isfinite(start) and math.isfinite(end) and start >= 0
                     and abs((end - start) - length / 15) < 1e-4,
                     f"Episode {i} video duration does not match its frame count")
    rows = []
    columns = ["index", "episode_index", "frame_index", "timestamp", "task_index", "observation.state", "action"]
    for path in sorted(data_files):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    _require(len(rows) == n_frames, "Data Parquet frame count disagrees with info.json")
    # Official loaders preserve file order. Reject shuffled or duplicated indices.
    for absolute_index, row in enumerate(rows):
        _require(row["index"] == absolute_index, "Data indices must be unique and in ascending file order")
    for i, episode in enumerate(episode_rows):
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        for frame, row in enumerate(rows[start:end]):
            _require(row["episode_index"] == i and row["frame_index"] == frame and row["task_index"] == 0,
                     f"Episode {i}, frame {frame}: invalid scalar indices")
            _require(math.isfinite(row["timestamp"]) and abs(row["timestamp"] - frame / 15) < 1e-4,
                     f"Episode {i}, frame {frame}: timestamp is not i/fps")
            for key in ("observation.state", "action"):
                vector = row[key]
                _require(len(vector) == 8 and all(math.isfinite(value) for value in vector),
                         f"Episode {i}, frame {frame}: {key} must be finite 8D")
                _require(-1e-6 <= vector[7] <= 0.080001,
                         f"Episode {i}, frame {frame}: {key} width outside 0..0.08 m")
    stats = _read_json(root / "meta/stats.json")
    for key in CAMERA_KEYS:
        _require(key in stats and all(name in stats[key] for name in ("min", "max", "mean", "std")),
                 f"Missing image statistics for {key}")
        # Per-channel image stats use LeRobot's C x 1 x 1 layout. Their
        # official-writer sample count need not equal the scalar frame count.
        for name in ("min", "max", "mean", "std"):
            value = stats[key][name]
            _require(isinstance(value, list) and len(value) == 3
                     and all(isinstance(channel, list) and len(channel) == 1
                             and isinstance(channel[0], list) and len(channel[0]) == 1
                             and isinstance(channel[0][0], (int, float))
                             and math.isfinite(channel[0][0]) for channel in value),
                     f"{key} {name} must be finite C x 1 x 1 statistics")
            if name == "std":
                _require(all(channel[0][0] >= 0 for channel in value), f"{key} std cannot be negative")
    return {
        "dataset_root": str(root), "info": info, "manifest": manifest, "fps": 15,
        "num_episodes": n_episodes, "num_frames": n_frames, "task": TASK,
        "camera_keys": CAMERA_KEYS.copy(), "feature_order": {"observation.state": STATE_NAMES.copy(), "action": ACTION_NAMES.copy()},
        "action_semantics": ACTION_SEMANTICS, "training_episodes": list(range(n_episodes)),
        "ignored_split_sidecar": (root / "meta/episode_splits.json").is_file(),
        "stats_scope": "all_training_episodes_recomputed_before_processor_creation",
    }


def make_dataset(dataset_root: str | Path, repo_id: str = "local/fr3_avocado_pi05", chunk_size: int = 50):
    """Load all local episodes using the official reader and its boundary padding.

    Actions are [t, t+1/fps, ..., t+(chunk_size-1)/fps]. At an episode's
    end, LeRobot repeats that episode's final action and sets action_is_pad;
    it never reads an action from the next episode. No manual data padding or
    action shifting is performed here.
    """
    _require(type(chunk_size) is int and chunk_size > 0, "chunk_size must be a positive integer")
    contract = inspect_contract(dataset_root)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=repo_id, root=Path(dataset_root).expanduser().resolve(), episodes=None,
        delta_timestamps={"action": [i / contract["fps"] for i in range(chunk_size)]},
        video_backend="pyav", download_videos=False, force_cache_sync=False,
    )
    _require(len(dataset) == contract["num_frames"], "Official reader did not load every training frame")
    return dataset


build_dataset = make_dataset


def stats_to_json(stats: Any) -> Any:
    """Convert nested NumPy/Torch statistics to plain JSON-compatible values."""
    if isinstance(stats, dict):
        return {str(key): stats_to_json(value) for key, value in stats.items()}
    if isinstance(stats, (list, tuple)):
        return [stats_to_json(value) for value in stats]
    if hasattr(stats, "detach"):
        return stats_to_json(stats.detach().cpu().tolist())
    if hasattr(stats, "tolist"):
        return stats_to_json(stats.tolist())
    if isinstance(stats, float):
        _require(math.isfinite(stats), "Statistics must be finite")
    if isinstance(stats, (str, int, float, bool)) or stats is None:
        return stats
    raise TypeError(f"Cannot serialize statistic of type {type(stats).__name__}")


def stats_to_tensors(stats: dict[str, Any]) -> dict[str, Any]:
    """Return CPU float32 tensors suitable for the LeRobot normalization processors."""
    import torch

    values = stats_to_json(stats)
    return {key: {name: torch.as_tensor(value, dtype=torch.float32) for name, value in feature.items()}
            for key, feature in values.items()}


def compute_training_stats(dataset) -> dict[str, Any]:
    """Compute exact statistics from every *original* state/action row once.

    In particular, this does not average episode statistics, use converter
    train/validation sidecars, or count repeated action-chunk padding. Image
    statistics are retained from the official writer's all-episode meta/stats.
    Returned values are plain lists/numbers and can be saved as JSON.
    """
    import numpy as np

    _require(getattr(dataset, "episodes", None) is None,
             "Training statistics require the full all-episode dataset (episodes=None)")
    root = Path(dataset.root).resolve()
    original = _read_json(root / "meta/stats.json")
    stats = {key: stats_to_json(original[key]) for key in CAMERA_KEYS}
    hf_data = dataset.hf_dataset.with_format("numpy")
    for key in ("observation.state", "action"):
        values = np.asarray(hf_data[key], dtype=np.float64)
        _require(values.shape == (len(dataset), 8) and len(values) > 0 and np.isfinite(values).all(),
                 f"{key} is not a complete, finite N x 8 scalar column")
        quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
        stats[key] = {
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0, ddof=0).tolist(),
            "count": [len(values)],
            **{name: row.tolist() for name, row in zip(("q01", "q10", "q50", "q90", "q99"), quantiles)},
        }
    return stats_to_json(stats)
