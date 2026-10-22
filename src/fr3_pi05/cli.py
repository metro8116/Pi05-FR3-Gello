"""Friendly offline CLI for the unmodified LeRobot 0.6.1 training entry point."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "steps": 20000, "batch_size": 16, "gradient_accumulation_steps": 1,
    "num_workers": 8, "save_freq": 2000, "log_freq": 10, "seed": 1000,
    "dtype": "bfloat16", "device": "cuda", "gradient_checkpointing": False,
    "freeze_vision_encoder": True, "train_expert_only": False,
    "chunk_size": 50, "n_action_steps": 50, "lr": 2.5e-5,
    "weight_decay": 0.01, "grad_clip_norm": 1.0,
    "wandb": False, "wandb_project": "fr3-avocado-pi05",
}
POLICY_FIELDS = {
    "dtype": "dtype", "device": "device", "gradient_checkpointing": "gradient_checkpointing",
    "freeze_vision_encoder": "freeze_vision_encoder", "train_expert_only": "train_expert_only",
    "chunk_size": "chunk_size", "n_action_steps": "n_action_steps", "lr": "optimizer_lr",
    "weight_decay": "optimizer_weight_decay", "grad_clip_norm": "optimizer_grad_clip_norm",
}
# Resume the saved experiment. Only operational settings may be changed here.
RESUME_FIXED = set(DEFAULTS) - {"num_workers", "save_freq", "log_freq", "device", "wandb", "wandb_project"}


def bool_arg(value: str) -> bool:
    if value.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("Use true or false")
    return value.lower() == "true"


def offline_environment() -> None:
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FR3 pi05_base fine-tuning through the official LeRobot 0.6.1 trainer")
    ap.add_argument("--dataset", "--data-root", dest="dataset", help="Local v3.0 dataset; default data/lerobot/fr3_avocado_v30")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--base-model", "--policy-path", dest="base_model", help="Local pi05_base snapshot or existing cached Hub ID")
    ap.add_argument("--tokenizer", help="Existing PaliGemma tokenizer directory or cached ID")
    ap.add_argument("--output-dir", "--out", dest="output_dir")
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--resume", nargs="?", const=True, default=False, help="Resume native checkpoint; optionally supply its path")
    ap.add_argument("--resume-from", help="Native checkpoint, pretrained_model directory, or train_config.json")
    for key in ("steps", "batch_size", "gradient_accumulation_steps", "num_workers", "save_freq", "log_freq", "seed", "chunk_size", "n_action_steps"):
        ap.add_argument("--" + key.replace("_", "-"), type=int, default=None)
    for key in ("lr", "weight_decay", "grad_clip_norm"):
        ap.add_argument("--" + key.replace("_", "-"), type=float, default=None)
    for key in ("gradient_checkpointing", "freeze_vision_encoder", "train_expert_only", "wandb"):
        ap.add_argument("--" + key.replace("_", "-"), type=bool_arg, default=None)
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default=None)
    ap.add_argument("--device", choices=("cuda", "cuda:0", "cpu"), default=None)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--offline", action="store_true", help="Compatibility flag; local files/cache only is always enabled")
    ap.add_argument("--dry-run", action="store_true", help="Print native arguments without writes, model imports, or training")
    return ap


def native_resume_config(candidate: str | Path) -> tuple[Path, dict]:
    path = Path(candidate).expanduser().resolve()
    if path.is_dir():
        if (path / "COMPLETE.json").exists() or (path / "training_state/state.pt").exists():
            raise ValueError("This is an old custom-loop checkpoint, not an official LeRobot checkpoint. Start a new native run.")
        if (path / "pretrained_model").is_dir():
            path = path / "pretrained_model"
        path = path / "train_config.json"
    checkpoint = path.parent.parent
    required = [path, *[path.parent / name for name in (
        "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")],
        *[checkpoint / "training_state" / name for name in (
            "optimizer_param_groups.json", "optimizer_state.safetensors", "rng_state.safetensors",
            "scheduler_state.json", "training_step.json")]]
    for item in required:
        if not item.is_file():
            raise ValueError(f"Not a complete official LeRobot checkpoint: missing {item}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    policy = saved.get("policy") or {}
    expected_images = {"observation.images.camera_1", "observation.images.camera_2"}
    features = policy.get("input_features", {})
    if (policy.get("type") != "pi05" or features.get("observation.state", {}).get("shape") != [8]
            or set(features) != expected_images | {"observation.state"}
            or any(features[key].get("shape") != [3, 480, 640] for key in expected_images)
            or policy.get("output_features", {}).get("action", {}).get("shape") != [8]
            or policy.get("use_relative_actions", False)):
        raise ValueError("Resume requires a native FR3 8D state/8D absolute-action, two-camera pi05 checkpoint.")
    dataset = saved.get("dataset") or {}
    if (dataset.get("episodes") is not None or dataset.get("eval_split", 0) != 0
            or dataset.get("streaming", False) or saved.get("peft") is not None):
        raise ValueError("Resume requires the complete non-streaming FR3 dataset without a validation split or PEFT.")
    # Validate state files referenced by the official processor configurations too.
    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = json.loads((path.parent / filename).read_text())
        for step in processor.get("steps", []):
            state_file = step.get("state_file")
            if state_file and not (path.parent / state_file).is_file():
                raise ValueError(f"Missing native processor state: {path.parent / state_file}")
    return path, saved


def resolve_config(args: argparse.Namespace) -> dict:
    saved, config_path = {}, None
    if args.resume or args.resume_from:
        if isinstance(args.resume, str) and args.resume_from and Path(args.resume).expanduser().resolve() != Path(args.resume_from).expanduser().resolve():
            raise ValueError("Use one checkpoint path for --resume and --resume-from")
        candidate = args.resume_from or (args.resume if isinstance(args.resume, str) else None)
        candidate = candidate or Path(args.output_dir or PROJECT_ROOT / "outputs/fr3_avocado_pi05") / "checkpoints/last"
        config_path, saved = native_resume_config(candidate)
        if args.base_model or args.tokenizer:
            raise ValueError("Native resume restores checkpoint weights/processors; omit --base-model/--policy-path/--tokenizer.")
    config = {"format": "fr3_pi05_native_v2", "trainer": "lerobot.scripts.lerobot_train.train", "required_lerobot": "0.6.1"}
    policy, wandb = saved.get("policy") or {}, saved.get("wandb") or {}
    explicit = {}
    for key, default in DEFAULTS.items():
        value = getattr(args, key)
        previous = (policy.get(POLICY_FIELDS[key], default) if key in POLICY_FIELDS else
                    wandb.get("enable" if key == "wandb" else "project", default) if key in {"wandb", "wandb_project"} else
                    saved.get(key, default))
        if config_path and key in RESUME_FIXED and value is not None and value != previous:
            raise ValueError(f"Native continuation preserves {key}={previous!r}; use a new run to change this setting.")
        config[key] = value if value is not None else previous
        if value is not None:
            explicit[key] = value
    dataset = args.dataset or saved.get("dataset", {}).get("root") or os.environ.get("PI05_DATASET") or PROJECT_ROOT / "data/lerobot/fr3_avocado_v30"
    config["dataset"] = str(Path(dataset).expanduser().resolve())
    config["output_dir"] = str(Path(args.output_dir or saved.get("output_dir") or PROJECT_ROOT / "outputs/fr3_avocado_pi05").expanduser().resolve())
    config["base_model"] = args.base_model or os.environ.get("PI05_BASE_MODEL") or (None if config_path else "lerobot/pi05_base")
    config["tokenizer"] = args.tokenizer or os.environ.get("PI05_TOKENIZER") or "google/paligemma-3b-pt-224"
    config["repo_id"] = args.repo_id or saved.get("dataset", {}).get("repo_id") or "local/fr3_avocado_pi05"
    config["job_name"] = args.job_name or saved.get("job_name") or "fr3_avocado_pi05"
    config["resume"] = str(config_path.parent.parent) if config_path else None
    config["config_path"] = str(config_path) if config_path else None
    config["explicit"] = explicit
    for key in ("steps", "batch_size", "save_freq", "log_freq", "chunk_size", "n_action_steps"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["num_workers"] < 0 or config["seed"] < 0:
        raise ValueError("num_workers and seed must be nonnegative")
    if config["gradient_accumulation_steps"] != 1:
        raise ValueError("LeRobot 0.6.1 official trainer does not support gradient accumulation; use 1.")
    if (not all(math.isfinite(config[key]) for key in ("lr", "grad_clip_norm", "weight_decay"))
            or config["lr"] <= 0 or config["grad_clip_norm"] <= 0 or config["weight_decay"] < 0):
        raise ValueError("lr and grad_clip_norm must be positive; weight_decay must be nonnegative")
    if config["n_action_steps"] > config["chunk_size"]:
        raise ValueError("n_action_steps cannot exceed chunk_size")
    config.update(validation=False, episode_selection="all", optimizer_betas=[0.9, 0.95], optimizer_eps=1e-8)
    return config


def native_arguments(config: dict, prepared_dir: str | Path | None = None) -> list[str]:
    def flag(name, value):
        if isinstance(value, bool):
            value = str(value).lower()
        return f"--{name}={value}"
    flags = [flag("dataset.root", config["dataset"]), "--dataset.video_backend=pyav",
             "--dataset.streaming=false", "--dataset.episodes=null", "--dataset.use_imagenet_stats=false", "--dataset.eval_split=0.0",
             "--eval_steps=0", "--env_eval_freq=0", "--use_policy_training_preset=true", "--save_checkpoint=true", "--save_checkpoint_to_hub=false",
             "--policy.push_to_hub=false", "--wandb.disable_artifact=true",
             flag("wandb.enable", config["wandb"]), flag("wandb.project", config["wandb_project"]),
             flag("output_dir", config["output_dir"])]
    if config["resume"]:
        flags += [flag("config_path", config["config_path"]), "--resume=true"]
        for key in ("num_workers", "save_freq", "log_freq", "device"):
            if key in config["explicit"]:
                flags.append(flag("policy.device" if key == "device" else key, config[key]))
        return flags
    flags += [flag("dataset.repo_id", config["repo_id"]), flag("job_name", config["job_name"]),
              flag("policy.path", prepared_dir or "<prepared_FR3_policy>")]
    flags += [flag(key, config[key]) for key in ("steps", "batch_size", "num_workers", "save_freq", "log_freq", "seed")]
    flags += [flag("policy." + name, config[key]) for key, name in POLICY_FIELDS.items()]
    flags += ["--policy.use_relative_actions=false", "--policy.optimizer_betas=[0.9,0.95]",
              "--policy.optimizer_eps=1e-08", "--policy.scheduler_warmup_steps=1000",
              "--policy.scheduler_decay_steps=30000", "--policy.scheduler_decay_lr=2.5e-06"]
    return flags


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        offline_environment()
        config = resolve_config(args)
        print(json.dumps({key: value for key, value in config.items() if key != "explicit"}, indent=2, ensure_ascii=False), flush=True)
        print("Official LeRobot arguments:\n" + shlex.join(["lerobot_train", *native_arguments(config)]), flush=True)
        if args.dry_run:
            print("Configuration only; no model loaded, training started, or output created.")
            return 0
        from .native import launch
        launch(config)
        print(f"Native export: {config['output_dir']}/checkpoints/last/pretrained_model", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Resume from a saved official checkpoint.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
