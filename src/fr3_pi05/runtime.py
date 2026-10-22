"""Local-only π0.5 model and processor adapters.

LeRobot 0.6.1 is the AutoDL baseline. 0.4.4 is retained as an explicitly
detected legacy interface; this module never installs or downloads anything.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Set these before importing Transformers / huggingface_hub. Tokenizer IDs may
# resolve from an existing cache, but are never fetched by the training code.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SUPPORTED_LEROBOT = {"0.6.1", "0.4.4"}
CAMERA_KEYS = ("observation.images.camera_1", "observation.images.camera_2")
PREPROCESSOR_NAME = "policy_preprocessor.json"
POSTPROCESSOR_NAME = "policy_postprocessor.json"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_runtime(require_cuda: bool = True) -> dict[str, Any]:
    """Check the selected existing environment without altering it."""
    versions = {
        name: _version(name)
        for name in ("lerobot", "torch", "transformers", "safetensors", "datasets", "av", "accelerate")
    }
    version = versions["lerobot"]
    if version not in SUPPORTED_LEROBOT:
        raise RuntimeError(
            f"Selected Python has lerobot={version!r}; supported interfaces are 0.6.1 "
            "(your existing AutoDL baseline) and legacy 0.4.4. Select the existing "
            "Pi05-Mujoco uv environment first. No environment has been changed."
        )
    missing = [name for name, value in versions.items() if value is None]
    if missing:
        raise RuntimeError(f"Selected Python is missing training dependencies: {', '.join(missing)}")
    try:
        import torch
        from accelerate import Accelerator
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
        from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "The selected environment cannot import the official π0.5 training API. "
            "Use the existing AutoDL uv Python; this code does not patch Transformers. "
            f"Original error: {exc}"
        ) from exc
    required_fields = {"max_state_dim", "max_action_dim", "input_features", "output_features"}
    if not required_fields.issubset(PI05Config.__dataclass_fields__):
        raise RuntimeError("The installed PI05Config is incompatible with this training adapter")
    if not hasattr(PI05Policy, "_fix_pytorch_state_dict_keys"):
        raise RuntimeError("The installed PI05Policy has no supported checkpoint-key adapter")
    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is unavailable in the selected Python; no training was started")
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "versions": versions,
        "lerobot_profile": "autodl_0.6.1" if version == "0.6.1" else "legacy_0.4.4",
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else False,
        "local_files_only": True,
    }


def _local_model_directory(base_dir: str | Path) -> Path:
    directory = Path(base_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {directory}")
    for name in ("config.json", "model.safetensors"):
        if not (directory / name).is_file():
            raise FileNotFoundError(
                f"Missing {directory / name}. Supply the local pi05_base snapshot containing "
                "config.json and one model.safetensors; sharded weights are not supported."
            )
    with (directory / "config.json").open(encoding="utf-8") as handle:
        raw_config = json.load(handle)
    if raw_config.get("type") != "pi05":
        raise ValueError(f"Expected a pi05 checkpoint, got type={raw_config.get('type')!r}")
    if raw_config.get("use_peft", False):
        raise ValueError("Supply full local pi05_base weights, not a PEFT-only adapter")
    return directory


def _same_tensor(left: Any, right: Any) -> bool:
    import torch

    return left.shape == right.shape and torch.equal(left, right)


def _complete_shared_aliases(policy: Any, state: dict[str, Any]) -> list[str]:
    """Fill only aliases of the same live Parameter/buffer, never missing layers.

    safetensors.save_model deliberately stores one member of tied weight groups.
    A missing independent parameter must still fail strict load_state_dict.
    """
    groups: dict[int, list[str]] = {}
    for iterator_name in ("named_parameters", "named_buffers"):
        iterator = getattr(policy, iterator_name)
        for name, value in iterator(remove_duplicate=False):
            groups.setdefault(id(value), []).append(name)
    filled: list[str] = []
    for names in groups.values():
        names = list(dict.fromkeys(names))
        present = [name for name in names if name in state]
        if not present:
            continue
        representative = state[present[0]]
        for name in present[1:]:
            if not _same_tensor(representative, state[name]):
                raise RuntimeError(f"Conflicting values for tied checkpoint weights: {present}")
        for name in names:
            if name not in state:
                state[name] = representative
                filled.append(name)
    return filled


def load_weights_strict(policy: Any, weights_path: str | Path) -> dict[str, Any]:
    """Migrate base-model keys and load strictly; never use for native resume.

    Initial pi05_base weights need LeRobot's architecture-key adapter. Native
    trainer exports already use the live model names and must instead pass to
    ``load_native_weights_strict`` so independent trained weights stay intact.
    """
    from safetensors.torch import load_file

    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    original = load_file(str(path), device="cpu")
    if not original:
        raise RuntimeError(f"The checkpoint contains no tensors: {path}")
    unprefixed: dict[str, Any] = {}
    for name, tensor in original.items():
        short_name = name.removeprefix("model.")
        if short_name in unprefixed:
            raise RuntimeError(f"Ambiguous model-prefix collision in checkpoint: {name}")
        unprefixed[short_name] = tensor
    fixed = policy._fix_pytorch_state_dict_keys(unprefixed, policy.config)
    remapped: dict[str, Any] = {}
    for name, tensor in fixed.items():
        target = name if name.startswith("model.") else f"model.{name}"
        if target in remapped and not _same_tensor(remapped[target], tensor):
            raise RuntimeError(f"Conflicting remapped checkpoint key: {target}")
        remapped[target] = tensor
    aliases = _complete_shared_aliases(policy, remapped)
    # The official PI05Policy.from_pretrained in both supported versions catches
    # loading errors. Bypass that method so wrong/incomplete base weights abort.
    result = policy.load_state_dict(remapped, strict=True)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"Incomplete checkpoint load: missing={missing}, unexpected={unexpected}")
    return {
        "format": "base_migration",
        "weights_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "source_tensor_count": len(original),
        "loaded_tensor_count": len(remapped),
        "key_remap_applied": set(original) != set(remapped),
        "shared_aliases_filled": aliases,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict": True,
    }


def _native_shared_live_groups(policy: Any, expected_keys: set[str]) -> list[list[str]]:
    """Find aliases of identical live objects; exclude nonpersistent buffers.

    Sharing storage alone does not make independent Parameters interchangeable.
    Only names present in state_dict() belong to this serialized schema.
    """
    groups: dict[int, list[str]] = {}
    for iterator_name in ("named_parameters", "named_buffers"):
        for name, value in getattr(policy, iterator_name)(remove_duplicate=False):
            if name in expected_keys:
                groups.setdefault(id(value), []).append(name)
    return [list(dict.fromkeys(names)) for names in groups.values() if len(names) > 1]


def load_native_weights_strict(policy: Any, weights_path: str | Path) -> dict[str, Any]:
    """Restore an official trainer export with its original keys and values.

    No model-prefix stripping, base migration or loader fallback is allowed.
    safetensors.save_model may omit a second name of the *same* live Parameter
    or persistent buffer; only those exact aliases can be completed. Validate
    keys, alias consistency and all shapes before copying any weight.
    """
    import torch
    from safetensors.torch import load_file

    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    state = load_file(str(path), device="cpu")
    if not state:
        raise RuntimeError(f"The native checkpoint contains no tensors: {path}")
    source_count = len(state)
    expected_state = policy.state_dict()
    expected_keys = set(expected_state)
    unexpected = sorted(set(state) - expected_keys)
    if unexpected:
        raise RuntimeError(
            "Native checkpoint has unexpected keys; no prefix/key migration is applied: "
            f"{unexpected}"
        )

    filled: list[str] = []
    for names in _native_shared_live_groups(policy, expected_keys):
        present = [name for name in names if name in state]
        if not present:
            continue
        representative = state[present[0]]
        for name in present[1:]:
            candidate = state[name]
            if (representative.dtype != candidate.dtype or representative.shape != candidate.shape
                    or not torch.equal(representative, candidate)):
                raise RuntimeError(f"Conflicting values for tied native checkpoint weights: {present}")
        for name in names:
            if name not in state:
                state[name] = representative
                filled.append(name)

    missing = sorted(expected_keys - set(state))
    if missing:
        raise RuntimeError(f"Native checkpoint is missing independent weights/buffers: {missing}")
    for name, tensor in state.items():
        target = expected_state[name]
        if not isinstance(target, torch.Tensor) or tensor.shape != target.shape:
            raise RuntimeError(
                f"Native checkpoint shape/type mismatch for {name}: "
                f"checkpoint={tuple(tensor.shape)}, model={getattr(target, 'shape', type(target).__name__)}"
            )

    result = policy.load_state_dict(state, strict=True)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(f"Incomplete native checkpoint load: missing={missing}, unexpected={unexpected}")
    return {
        "format": "native_state_dict",
        "weights_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "source_tensor_count": source_count,
        "loaded_tensor_count": len(state),
        "key_remap_applied": False,
        "shared_aliases_filled": filled,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict": True,
    }


def _fr3_features() -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.configs.types import FeatureType, PolicyFeature

    inputs = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))}
    inputs.update(
        {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for key in CAMERA_KEYS}
    )
    return inputs, {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}


def _load_tokenizer(tokenizer_source: str | Path) -> Any:
    from transformers import AutoTokenizer

    # Preserve a cached Hub ID such as google/paligemma-3b-pt-224, while
    # expanding a user-specified local ~/... directory.
    source = str(Path(str(tokenizer_source)).expanduser())
    try:
        return AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise RuntimeError(
            f"Tokenizer is not available locally: {source}. Point PI05_TOKENIZER at the existing "
            "PaliGemma tokenizer directory or keep its existing Hugging Face cache. "
            "No download was attempted."
        ) from exc


def make_processors(
    policy_cfg: Any,
    stats: dict[str, Any],
    tokenizer_source: str | Path,
    device: str,
) -> tuple[Any, Any]:
    """Use the official PI05 processor factory with a standard local tokenizer.

    This changes only the factory's hard-coded tokenizer source during initial
    construction. Every returned step remains an official LeRobot step. Native
    training, processor serialization and checkpoint saving remain untouched.
    """
    module = importlib.import_module("lerobot.policies.pi05.processor_pi05")

    for feature in ("observation.state", "action"):
        if feature not in stats or not {"q01", "q99"}.issubset(stats[feature]):
            raise ValueError(f"Current-task quantile statistics are required for {feature}")
    if getattr(policy_cfg, "use_relative_actions", False):
        raise ValueError("This dataset uses absolute actions; use_relative_actions must be False")
    tokenizer = _load_tokenizer(tokenizer_source)
    source = str(Path(str(tokenizer_source)).expanduser())
    if Path(source).is_dir():
        source = str(Path(source).resolve())
    original_step = module.TokenizerProcessorStep

    def standard_local_tokenizer(*args, **kwargs):
        kwargs.update(tokenizer_name=source, tokenizer=tokenizer)
        step = original_step(*args, **kwargs)
        # Preserve the cached source in the standard step's get_config().
        step.tokenizer = None
        return step

    original_device = policy_cfg.device
    try:
        policy_cfg.device = device
        module.TokenizerProcessorStep = standard_local_tokenizer
        return module.make_pi05_pre_post_processors(config=policy_cfg, dataset_stats=stats)
    finally:
        module.TokenizerProcessorStep = original_step
        policy_cfg.device = original_device


def prepare_native_policy(
    base_dir: str | Path,
    prepared_dir: str | Path,
    *,
    stats: dict[str, Any],
    tokenizer_source: str | Path,
    policy_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Prepare ``--policy.path`` without instantiating or duplicating the model.

    The standard tokenizer step saves its source ID/path, matching the official
    trainer. Moving a checkpoint therefore requires the same cached PaliGemma
    tokenizer ID, or the explicit stable local tokenizer path; it is not bundled.
    """
    from safetensors import SafetensorError, safe_open
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import NormalizationMode
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from .data import stats_to_tensors

    base = _local_model_directory(base_dir)
    destination = Path(prepared_dir).expanduser().resolve()
    if destination == base or destination.is_relative_to(base) or base.is_relative_to(destination):
        raise ValueError("Prepared policy directory must be separate from the original base model")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Prepared policy output already exists: {destination}")
    weights = base / "model.safetensors"
    # Only read the safetensors header, never materialize a model or its weights.
    try:
        with safe_open(str(weights), framework="pt", device="cpu") as header:
            tensor_count = len(list(header.keys()))
    except SafetensorError as exc:
        raise ValueError(f"Invalid local safetensors header: {weights}: {exc}") from exc
    if tensor_count == 0:
        raise ValueError("Local base safetensors contains no tensor entries")
    config = PreTrainedConfig.from_pretrained(base, local_files_only=True)
    reserved = {"input_features", "output_features", "max_state_dim", "max_action_dim", "empty_cameras",
                "n_obs_steps", "use_relative_actions", "normalization_mapping", "use_peft", "pretrained_path"}
    for key, value in policy_overrides.items():
        if key in reserved or key not in config.__dataclass_fields__:
            raise ValueError(f"Unsupported native policy override (FR3 interface fields are fixed): {key}")
        setattr(config, key, value)
    config.input_features, config.output_features = _fr3_features()
    config.max_state_dim = config.max_action_dim = 32
    config.empty_cameras = 0
    config.n_obs_steps = 1
    config.use_peft = False
    config.pretrained_path = None  # TrainPipelineConfig sets this from --policy.path.
    config.push_to_hub = False
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,
        "ACTION": NormalizationMode.QUANTILES,
    }
    if hasattr(config, "use_relative_actions"):
        config.use_relative_actions = False
        config.action_feature_names = [f"fr3_joint{i}" for i in range(1, 8)] + ["gripper_target_width_m"]
    if not 1 <= config.n_action_steps <= config.chunk_size:
        raise ValueError("Require 1 <= n_action_steps <= chunk_size")
    config.validate_features()
    pre, post = make_processors(config, stats_to_tensors(stats), tokenizer_source, config.device)
    source = str(Path(str(tokenizer_source)).expanduser())
    local_source = Path(source).is_dir()
    if local_source:
        source = str(Path(source).resolve())
    report = {
        "prepared_dir": str(destination), "base_dir": str(base), "weights_source": str(weights.resolve()),
        "weight_link": "symbolic_link", "header_tensor_count": tensor_count,
        "large_model_allocated": False, "weights_duplicated": False,
        "full_strict_load": "performed by the native trainer at policy construction",
        "state_dim": 8, "action_dim": 8, "camera_keys": list(CAMERA_KEYS),
        "relative_actions": False, "processor_factory": "official make_pi05_pre_post_processors",
        "tokenizer": {
            "source": source, "bundled": False, "custom_processor_class": False,
            "checkpoint_requirement": "Keep this stable local path on the target machine" if local_source
            else "The target machine must already cache this exact tokenizer ID (offline loading)",
        },
        "native_loop_sampler_optimizer_checkpoint_modified": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    try:
        (destination / "model.safetensors").symlink_to(weights.resolve())
        config.save_pretrained(destination)
        # Legacy 0.4.4 serialization can omit the registry tag. Native 0.6.1
        # already writes it; preserve this reader compatibility for CPU tests.
        config_path = destination / "config.json"
        serialized = json.loads(config_path.read_text(encoding="utf-8"))
        if "type" not in serialized:
            serialized["type"] = "pi05"
            config_path.write_text(json.dumps(serialized, indent=2) + "\n", encoding="utf-8")
        pre.save_pretrained(destination, config_filename=PREPROCESSOR_NAME)
        post.save_pretrained(destination, config_filename=POSTPROCESSOR_NAME)
        (destination / "fr3_preparation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(destination)
        raise
    return report


def install_strict_native_loader(*, checkpoint_format: str = "base"):
    """Replace only PI05's error-swallowing loader; return an undo callback.

    The official trainer still constructs the policy with its native config,
    then handles Accelerate, sampling, optimization and checkpointing itself.
    The caller must explicitly select native for resume; base remains the
    compatible default for initial pi05_base loading. Never infer by pathname.
    """
    if checkpoint_format not in {"base", "native"}:
        raise ValueError("checkpoint_format must be 'base' or 'native'")
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if getattr(PI05Policy, "_fr3_strict_native_loader", False):
        if getattr(PI05Policy, "_fr3_strict_checkpoint_format", None) != checkpoint_format:
            raise RuntimeError("A strict PI05 loader is already installed for a different checkpoint format")
        return lambda: None
    original = PI05Policy.__dict__["from_pretrained"]
    weight_loader = load_native_weights_strict if checkpoint_format == "native" else load_weights_strict

    def strict_from_pretrained(cls, pretrained_name_or_path, *, config=None, strict=True,
                               local_files_only=True, revision=None, **kwargs):
        directory = _local_model_directory(pretrained_name_or_path)
        if config is None:
            from lerobot.configs.policies import PreTrainedConfig
            config = PreTrainedConfig.from_pretrained(directory, local_files_only=True)
        # Preserve the supplied config, including dtype, freezing and checkpoint
        # options. Never call the old from_pretrained fallback implementation.
        policy = cls(config, **kwargs)
        report = weight_loader(policy, directory / "model.safetensors")
        policy._fr3_weight_load_report = report
        print(f"Strict local π0.5 load: {report['loaded_tensor_count']} tensors; "
              "missing=0, unexpected=0", flush=True)
        return policy

    installed = classmethod(strict_from_pretrained)
    PI05Policy.from_pretrained = installed
    PI05Policy._fr3_strict_native_loader = True
    PI05Policy._fr3_strict_checkpoint_format = checkpoint_format

    def restore():
        if PI05Policy.__dict__.get("from_pretrained") is installed:
            PI05Policy.from_pretrained = original
            delattr(PI05Policy, "_fr3_strict_native_loader")
            delattr(PI05Policy, "_fr3_strict_checkpoint_format")

    return restore
