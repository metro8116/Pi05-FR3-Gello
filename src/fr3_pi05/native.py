"""Data/processor bootstrap; LeRobot owns every training and checkpoint operation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from .cli import native_arguments


def resolve_local_base(source: str) -> Path:
    """Resolve a directory or an already cached pi05_base ID without network access."""
    directory = Path(source).expanduser()
    if directory.is_dir():
        return directory.resolve()
    if source != "lerobot/pi05_base":
        raise FileNotFoundError(f"Local base model directory does not exist: {directory}")
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=source, local_files_only=True)).resolve()
    except Exception as exc:
        raise RuntimeError("pi05_base is not in the selected Hugging Face cache. Set PI05_BASE_MODEL to the existing snapshot directory containing config.json and model.safetensors.") from exc


def read_task_stats(root: str | Path, num_frames: int) -> dict[str, Any]:
    """Use the exported dataset statistics also consumed by the native trainer."""
    import math
    stats = json.loads((Path(root) / "meta/stats.json").read_text(encoding="utf-8"))
    for feature in ("observation.state", "action"):
        values = stats.get(feature, {})
        if values.get("count") != [num_frames]:
            raise ValueError(f"{feature} statistics must cover all {num_frames} frames")
        for name in ("mean", "std", "min", "max", "q01", "q99"):
            vector = values.get(name, [])
            if len(vector) != 8 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
                raise ValueError(f"Invalid exported {feature}/{name} statistics")
    return stats


def invoke_official(flags: list[str]) -> None:
    """Call the same decorated official entry point as the old project."""
    from .runtime import install_strict_native_loader
    # native_arguments emits this exact flag for official checkpoint recovery.
    # Native exports retain independent lm_head/embed_tokens weights; only a
    # fresh pi05_base initialization may run the base-key migration helper.
    checkpoint_format = "native" if "--resume=true" in flags else "base"
    restore = install_strict_native_loader(checkpoint_format=checkpoint_format)
    previous_argv = sys.argv
    try:
        from lerobot.scripts.lerobot_train import train
        sys.argv = ["lerobot_train", *flags]
        train()
    finally:
        sys.argv = previous_argv
        restore()


def launch(config: dict[str, Any]) -> None:
    from .data import inspect_contract, stats_to_tensors
    from .runtime import check_runtime, prepare_native_policy

    output = Path(config["output_dir"])
    if not config["resume"] and output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --out or resume an official checkpoint.")
    contract = inspect_contract(config["dataset"])
    stats = read_task_stats(config["dataset"], contract["num_frames"])
    runtime = check_runtime(require_cuda=config["device"] != "cpu")
    if runtime["versions"]["lerobot"] != "0.6.1":
        raise RuntimeError("Native training requires the existing AutoDL LeRobot 0.6.1 environment. Local legacy 0.4.4 is supported only for data/processor inspection; no environment was changed.")
    if config["device"] != "cpu" and config["dtype"] == "bfloat16" and not runtime["bf16_supported"]:
        raise RuntimeError("The selected CUDA device does not support bfloat16")
    print(f"Official training: all {contract['num_episodes']} episodes / {contract['num_frames']} frames, {contract['fps']} FPS; no validation.", flush=True)
    print(f"Python: {runtime['python']}; LeRobot: {runtime['versions']['lerobot']}", flush=True)
    prepared = None
    if not config["resume"]:
        base = resolve_local_base(config["base_model"])
        output.parent.mkdir(parents=True, exist_ok=True)
        # A sibling of output: native TrainPipelineConfig must create its own output directory.
        prepared = Path(tempfile.mkdtemp(prefix=f".{output.name}_prepared_", dir=output.parent)) / "policy"
        overrides = {key: config[key] for key in (
            "device", "dtype", "chunk_size", "n_action_steps", "gradient_checkpointing",
            "freeze_vision_encoder", "train_expert_only")}
        report = prepare_native_policy(base, prepared, stats=stats_to_tensors(stats),
                                       tokenizer_source=config["tokenizer"], policy_overrides=overrides)
        print(json.dumps({"fr3_policy_adapter": report}, ensure_ascii=False, indent=2), flush=True)
    # No model optimizer, loader, sampler, RNG cursor, save routine, or update loop here.
    invoke_official(native_arguments(config, prepared))
