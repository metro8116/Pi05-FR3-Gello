"""Offline read-only preflight, without allocating the π0.5 network."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from .cli import PROJECT_ROOT, offline_environment


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Check all local FR3 avocado episodes, existing uv environment, and local π0.5 assets"
    )
    ap.add_argument("--dataset", default=os.environ.get("PI05_DATASET") or str(PROJECT_ROOT / "data/lerobot/fr3_avocado_v30"))
    ap.add_argument("--base-model", default=os.environ.get("PI05_BASE_MODEL"))
    ap.add_argument("--tokenizer", default=os.environ.get("PI05_TOKENIZER") or "google/paligemma-3b-pt-224")
    ap.add_argument("--data-only", action="store_true", help="Check data without importing the model API or requiring base weights/CUDA")
    ap.add_argument("--cpu", action="store_true", help="Allow a CPU runtime inspection; this does not start CPU training")
    return ap


def _array(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _check_decoded_samples(dataset: Any, contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode representative frames and verify the official end padding."""
    import numpy as np
    from .data import CAMERA_KEYS, TASK

    count = contract["num_frames"]
    first_episode_end = int(dataset.meta.episodes[0]["dataset_to_index"])
    indices = sorted({0, first_episode_end - 1, min(first_episode_end, count - 1), count - 1})
    terminal_indices = {first_episode_end - 1, count - 1}
    reports = []
    for index in indices:
        sample = dataset[index]
        state, action = _array(sample["observation.state"]), _array(sample["action"])
        if state.shape != (8,) or action.shape != (50, 8):
            raise ValueError(f"Sample {index}: expected state (8,) and action (50,8); got {state.shape}, {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Sample {index}: non-finite state/action")
        current_action = _array(dataset.hf_dataset[index]["action"])
        if not np.array_equal(action[0], current_action):
            raise ValueError(f"Sample {index}: action chunk does not start at the current frame")
        if sample.get("task") != TASK:
            raise ValueError(f"Sample {index}: wrong task text")
        cameras = {}
        for key in CAMERA_KEYS:
            image = _array(sample[key])
            if image.shape != (3, 480, 640) or image.dtype.kind != "f" or not np.isfinite(image).all():
                raise ValueError(f"Sample {index}: invalid decoded image for {key}: {image.shape}")
            lo, hi = float(image.min()), float(image.max())
            if lo < 0 or hi > 1:
                raise ValueError(f"Sample {index}: RGB floats must be in [0,1] for {key}")
            cameras[key] = {"shape": list(image.shape), "range": [lo, hi]}
        report = {"index": index, "state_shape": [8], "action_shape": [50, 8], "cameras": cameras}
        if index in terminal_indices:
            expected = _array(dataset.hf_dataset[index]["action"])
            padded = _array(sample["action_is_pad"])
            if (padded.shape != (50,) or bool(padded[0]) or not padded[1:].all()
                    or not np.array_equal(action, np.repeat(expected[None, :], 50, axis=0))):
                raise ValueError(f"Sample {index}: end-of-episode action padding crosses a boundary or is invalid")
            report["end_padding_verified"] = True
        reports.append(report)
    return reports


def inspect_weight_header(base_model: str | Path) -> dict[str, Any]:
    """Inspect safetensors shapes, without loading weights or constructing a model."""
    from safetensors import SafetensorError, safe_open
    from .runtime import _local_model_directory

    directory = _local_model_directory(base_model)
    weights = directory / "model.safetensors"
    total_elements = 0
    try:
        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise ValueError(f"Checkpoint has no tensor entries: {weights}")
            for key in keys:
                shape = handle.get_slice(key).get_shape()
                if any(type(size) is not int or size <= 0 for size in shape):
                    raise ValueError(f"Invalid checkpoint tensor shape: {key}: {shape}")
                total_elements += math.prod(shape)
    except SafetensorError as exc:
        raise ValueError(f"Invalid local safetensors checkpoint: {weights}: {exc}") from exc
    return {
        "directory": str(directory),
        "weights_file": str(weights),
        "file_size_bytes": weights.stat().st_size,
        "tensor_count": len(keys),
        "total_tensor_elements": total_elements,
        "header_shapes_checked": True,
        "full_strict_weight_load": "not performed; checked when training loads the complete model",
    }


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    from .data import compute_training_stats, inspect_contract, make_dataset

    dataset_root = Path(args.dataset).expanduser().resolve()
    contract = inspect_contract(dataset_root)
    dataset = make_dataset(dataset_root, chunk_size=50)
    stats = compute_training_stats(dataset)
    scalar_counts = {key: stats[key]["count"] for key in ("observation.state", "action")}
    if any(value != [contract["num_frames"]] for value in scalar_counts.values()):
        raise ValueError("Scalar normalization statistics must cover every original training row exactly once")
    # The official trainer consumes meta/stats.json; ensure its statistics agree
    # with this independent all-row check rather than silently substituting ours.
    from .native import read_task_stats
    import numpy as np
    exported_stats = read_task_stats(dataset_root, contract["num_frames"])
    for key in ("observation.state", "action"):
        for name in ("mean", "std", "min", "max", "q01", "q99"):
            if not np.allclose(exported_stats[key][name], stats[key][name], rtol=1e-10, atol=1e-10):
                raise ValueError(f"Exported {key}/{name} statistics disagree with all training rows")
    samples = _check_decoded_samples(dataset, contract)
    summary = {
        "status": "passed",
        "dataset": {
            "root": str(dataset_root),
            "format": "v3.0",
            "episodes": contract["num_episodes"],
            "frames": contract["num_frames"],
            "fps": contract["fps"],
            "task": contract["task"],
            "episode_selection": "all",
            "validation": False,
            "old_split_sidecar_ignored": contract["ignored_split_sidecar"],
            "scalar_statistics_counts": scalar_counts,
            "exported_stats_verified": True,
            "scalar_quantiles": "q01/q10/q50/q90/q99 recomputed from all original rows",
            "decoded_samples": samples,
        },
        "training_defaults": {
            "gradient_checkpointing": False,
            "freeze_vision_encoder": True,
            "train_expert_only": False,
            "chunk_size": 50,
            "state_dim": 8,
            "action_dim": 8,
            "relative_actions": False,
        },
        "checks_only": True,
        "training_started": False,
        "large_model_allocated": False,
    }
    if args.data_only:
        summary["runtime"] = "skipped (--data-only)"
        summary["base_model"] = "skipped (--data-only)"
        summary["tokenizer"] = "skipped (--data-only)"
        return summary
    if not args.base_model:
        raise ValueError("Provide --base-model /actual/local/pi05_base or PI05_BASE_MODEL; use --data-only to check data alone")
    from .runtime import _load_tokenizer, check_runtime

    summary["runtime"] = check_runtime(require_cuda=not args.cpu)
    summary["base_model"] = inspect_weight_header(args.base_model)
    tokenizer = _load_tokenizer(args.tokenizer)
    summary["tokenizer"] = {
        "source": str(args.tokenizer),
        "available_locally": True,
        "vocabulary_size": int(len(tokenizer)),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    offline_environment()
    try:
        summary = run_checks(args)
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Preflight interrupted; no training was started.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
