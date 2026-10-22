"""Local-only PI05 loading and inference. No robot or dataset is opened here."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import struct
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from config import REPO_ROOT

# These must be set before importing Transformers, huggingface_hub or LeRobot.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from protocol import (CAMERA_KEYS, PROTOCOL_VERSION, TASK, decode_image,
                      validate_actions_response, validate_observation_request)

ACTION_SEMANTICS = "absolute_joint_positions_rad_and_total_gripper_width_m"
FPS = 15
CHUNK_SIZE = 50


class ServerBusyError(RuntimeError):
    """The model already has an in-flight request; never queue stale observations."""


def resolve_checkpoint(source: str | Path) -> Path:
    """Resolve ``last`` once so an ongoing training save cannot switch our model."""
    path = Path(source).expanduser().resolve(strict=True)
    if not (path / "config.json").is_file() and (path / "pretrained_model").is_dir():
        path = (path / "pretrained_model").resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Checkpoint must be a local pretrained_model directory: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint asset is missing: {path}")
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError(f"Checkpoint JSON is unexpectedly large: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Inspect only the header and offsets; do not read multi-GB tensor payloads."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", raw)[0]
        if not 2 <= header_size <= min(32 * 1024 * 1024, size - 8):
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(handle.read(header_size))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not tensors:
        raise ValueError(f"No tensors in checkpoint: {path}")
    data_size = size - header_size - 8
    for name, entry in tensors.items():
        offsets = entry.get("data_offsets", [])
        if (len(offsets) != 2 or any(type(x) is not int for x in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_size):
            raise ValueError(f"Invalid tensor offsets for {name}: {path}")
    return {"tensor_count": len(tensors), "file_size_bytes": size}


def inspect_checkpoint(source: str | Path) -> dict[str, Any]:
    """Validate the saved FR3 interface and every named processor state file."""
    path = resolve_checkpoint(source)
    config = _read_json(path / "config.json")
    if config.get("type") != "pi05" or config.get("use_peft", False):
        raise ValueError("A compatible full PI05 checkpoint is required (not a PEFT adapter)")
    inputs, outputs = config.get("input_features", {}), config.get("output_features", {})
    if set(inputs) != {"observation.state", *CAMERA_KEYS} or set(outputs) != {"action"}:
        raise ValueError("Checkpoint must use exactly the FR3 state, two named cameras and action")
    if inputs["observation.state"].get("shape") != [8] or outputs["action"].get("shape") != [8]:
        raise ValueError("Checkpoint state and action dimensions must both be 8")
    if any(inputs[key].get("shape") != [3, 480, 640] for key in CAMERA_KEYS):
        raise ValueError("Checkpoint cameras must both be RGB CHW [3,480,640]")
    if config.get("use_relative_actions", False):
        raise ValueError("This deployment expects absolute actions, not relative actions")
    if config.get("chunk_size") != CHUNK_SIZE or config.get("n_obs_steps", 1) != 1:
        raise ValueError("This deployment requires one observation and a 50-action chunk")
    normalization = config.get("normalization_mapping", {})
    if normalization.get("STATE") != "QUANTILES" or normalization.get("ACTION") != "QUANTILES":
        raise ValueError("Saved FR3 state/action normalization must be QUANTILES")
    processor_configs = {}
    state_assets = []
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        saved = _read_json(path / name)
        steps = saved.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Missing official processor steps: {name}")
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"Invalid official processor step: {name}")
            registry = step.get("registry_name", "")
            if registry in ("relative_actions_processor", "absolute_actions_processor"):
                if step.get("config", {}).get("enabled", False):
                    raise ValueError("Checkpoint processor enables relative actions")
            state_file = step.get("state_file")
            if state_file:
                state_path = (path / state_file).resolve()
                if state_path.parent != path:
                    raise ValueError("Processor state must be inside its checkpoint directory")
                _read_safetensors_header(state_path)
                state_assets.append(state_file)
        processor_configs[name] = saved
    for processor_name, expected_registry in (("policy_preprocessor.json", "normalizer_processor"),
                                               ("policy_postprocessor.json", "unnormalizer_processor")):
        normalized = [step for step in processor_configs[processor_name]["steps"]
                      if step.get("registry_name") == expected_registry]
        if len(normalized) != 1 or not normalized[0].get("state_file"):
            raise ValueError(f"Saved {expected_registry} statistics are required; never rebuild them from data")
    tokenizers = [step for step in processor_configs["policy_preprocessor.json"]["steps"]
                  if step.get("registry_name") == "tokenizer_processor"]
    if len(tokenizers) != 1 or not tokenizers[0].get("config", {}).get("tokenizer_name"):
        raise ValueError("Checkpoint must contain one saved official tokenizer processor")
    weights = path / "model.safetensors"
    header = _read_safetensors_header(weights)
    digest = hashlib.sha256()
    digest.update(str(path).encode())
    digest.update((path / "config.json").read_bytes())
    digest.update(f"{weights.stat().st_size}:{weights.stat().st_mtime_ns}".encode())
    return {"checkpoint": str(path), "config": config, "processor_configs": processor_configs,
            "processor_state_assets": state_assets, "model_id": "pi05-" + digest.hexdigest()[:20],
            "identity_method": "resolved_path_config_weight_size_mtime; not a full weights hash",
            "tokenizer": tokenizers[0]["config"]["tokenizer_name"], **header}


class OfficialPolicyRuntime:
    """Run the exact saved official preprocessor, model and postprocessor."""

    mock = False

    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any, *,
                 checkpoint: str, model_id: str, device: str, dtype: str,
                 torch_module: Any = None):
        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor
        self.checkpoint, self.model_id, self.device, self.dtype = checkpoint, model_id, device, dtype
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module

    def predict(self, request: dict[str, Any]) -> list[list[float]]:
        import numpy as np

        torch = self.torch
        observation = request["observation"]
        # No padding, quantile recomputation or manual action scaling. Saved
        # official processors perform exactly the transformations used in training.
        batch = {"observation.state": torch.tensor(observation["state"], dtype=torch.float32),
                 "task": observation["task"]}
        for key in CAMERA_KEYS:
            rgb = decode_image(observation["images"][key])
            pixels = np.ascontiguousarray(rgb.transpose(2, 0, 1))
            batch[key] = torch.from_numpy(pixels).to(dtype=torch.float32).div_(255.0)
        with torch.inference_mode():
            prepared = self.preprocessor(batch)
            # PI05 keeps vision/layernorm parameters in FP32 while its decoder
            # uses BF16. Match the native trainer's dtype-driven autocast, and
            # official control_utils.predict_action's CUDA AMP inference path.
            amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if self.device.startswith("cuda") and self.dtype == "bfloat16" else nullcontext())
            with amp:
                normalized = self.policy.predict_action_chunk(prepared)
            actions = self.postprocessor(normalized)
        if tuple(actions.shape) != (1, CHUNK_SIZE, 8):
            raise ValueError(f"PI05 returned unexpected action shape: {tuple(actions.shape)}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("PI05 returned NaN or infinite actions")
        return actions.detach().to(device="cpu", dtype=torch.float32)[0].tolist()


def load_official_runtime(checkpoint: str | Path, train_project: str | Path | None = None, *,
                          device: str = "cuda", dtype: str = "bfloat16",
                          tokenizer: str | None = None,
                          check_only: bool = False) -> OfficialPolicyRuntime | dict[str, Any]:
    """Use the existing AutoDL environment, without downloads or dataset access."""
    report = inspect_checkpoint(checkpoint)
    version = importlib.metadata.version("lerobot")
    if version != "0.6.1":
        raise RuntimeError(f"Deployment targets your AutoDL LeRobot 0.6.1; selected version is {version}")
    training_src = Path(train_project or REPO_ROOT).expanduser().resolve() / "src"
    if not (training_src / "fr3_pi05" / "runtime.py").is_file():
        raise FileNotFoundError(f"Strict PI05 loader is missing: {training_src / 'fr3_pi05/runtime.py'}")
    sys.path.insert(0, str(training_src))
    from fr3_pi05.runtime import _load_tokenizer, load_native_weights_strict
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import torch

    if dtype not in ("bfloat16", "float32"):
        raise ValueError("PI05 supports bfloat16 or float32, not float16")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected existing environment")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected GPU does not support bfloat16")
    elif device != "cpu":
        raise ValueError("Supported deployment devices are cuda[:N] or cpu")
    source = tokenizer or report["tokenizer"]
    # This catches cache/path problems before allocating the multi-GB policy.
    _load_tokenizer(source)
    path = Path(report["checkpoint"])
    cfg = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    cfg.device, cfg.dtype, cfg.gradient_checkpointing = device, dtype, False
    cfg.push_to_hub = False
    report.update(python=sys.executable, lerobot_version=version, device=device, dtype=dtype,
                  tokenizer=source, local_files_only=True, large_model_allocated=False)
    if check_only:
        return {key: value for key, value in report.items()
                if key not in ("config", "processor_configs")}
    pre_overrides = {"device_processor": {"device": device}}
    if tokenizer:
        pre_overrides["tokenizer_processor"] = {"tokenizer_name": source}
    # Critical: no dataset_stats override. Checkpoint processor statistics are
    # authoritative and independent of the training dataset's presence.
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(path), preprocessor_overrides=pre_overrides)
    policy = PI05Policy(cfg)
    # Official trainer exports already use this policy's native state_dict names.
    # The base-model migration helper aliases lm_head to embed_tokens and must
    # not be applied to a fine-tuned native checkpoint with distinct values.
    weight_report = load_native_weights_strict(policy, path / "model.safetensors")
    print(f"Strict PI05 load: {weight_report['loaded_tensor_count']} tensors; missing=0, unexpected=0", flush=True)
    policy.to(device)
    policy.eval()
    return OfficialPolicyRuntime(policy, pre, post, checkpoint=str(path),
                                 model_id=report["model_id"], device=device, dtype=dtype,
                                 torch_module=torch)


class MockPolicyRuntime:
    """Explicit communication-only simulator. Clients must prohibit robot arming."""

    mock = True
    checkpoint = "MOCK_NO_CHECKPOINT"
    model_id = "mock-no-robot-execution"
    device = "cpu"
    dtype = "float32"

    def predict(self, request: dict[str, Any]) -> list[list[float]]:
        # Decode both images even in mock mode to exercise the actual wire path.
        for key in CAMERA_KEYS:
            decode_image(request["observation"]["images"][key])
        return [list(request["observation"]["state"]) for _ in range(CHUNK_SIZE)]


class InferenceService:
    def __init__(self, runtime: OfficialPolicyRuntime | MockPolicyRuntime, task: str = TASK):
        self.runtime = runtime
        self.task = task
        self.model_lock = threading.Lock()

    def metadata(self) -> dict[str, Any]:
        return {"version": PROTOCOL_VERSION, "type": "metadata", "state_dim": 8, "action_dim": 8,
                "camera_keys": list(CAMERA_KEYS), "fps": FPS, "chunk_size": CHUNK_SIZE,
                "action_semantics": ACTION_SEMANTICS, "model_id": self.runtime.model_id,
                "checkpoint": self.runtime.checkpoint, "mock": self.runtime.mock,
                "device": self.runtime.device, "dtype": self.runtime.dtype, "task": self.task}

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        validate_observation_request(request)
        if request["observation"]["task"] != self.task:
            raise ValueError("Request task differs from this server's configured trained task")
        if not self.model_lock.acquire(blocking=False):
            raise ServerBusyError("PI05 is busy; discard this observation and retry with a fresh one")
        started = time.monotonic()
        try:
            actions = self.runtime.predict(request)
            response = {"version": PROTOCOL_VERSION, "type": "actions", "fps": FPS,
                        "actions": actions, "model_id": self.runtime.model_id,
                        "inference_s": time.monotonic() - started, "mock": self.runtime.mock,
                        **{key: request[key] for key in ("session_id", "request_id", "capture_seq",
                                                       "capture_monotonic")}}
            validate_actions_response(response, request=request, model_id=self.runtime.model_id)
            return response
        finally:
            self.model_lock.release()
