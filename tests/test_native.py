"""CPU orchestration checks: native training owns construction and its lifecycle."""

from __future__ import annotations

import builtins
from contextlib import ExitStack, redirect_stdout
from functools import wraps
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fr3_pi05 import data, native, runtime
from fr3_pi05.cli import parser, resolve_config

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


def statistics(frames=3):
    return {feature: {"count": [frames], "mean": [0.0] * 8, "std": [1.0] * 8,
                      "min": [-1.0] * 8, "max": [1.0] * 8,
                      "q01": [-0.9] * 8, "q99": [0.9] * 8}
            for feature in ("observation.state", "action")}


def assets(root):
    dataset = root / "data"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/stats.json").write_text(json.dumps(statistics()), encoding="utf-8")
    base = root / "existing_base"
    base.mkdir()
    config = resolve_config(parser().parse_args([
        "--data-root", str(dataset), "--policy-path", str(base),
        "--out", str(root / "new_run"), "--device", "cpu", "--dtype", "float32",
    ]))
    return config


def native_runtime(version="0.6.1"):
    return {"versions": {"lerobot": version}, "python": sys.executable, "bf16_supported": False}


class NativeEntryPointTests(unittest.TestCase):
    def entry_modules(self, entry):
        package = types.ModuleType("lerobot")
        package.__path__ = []
        scripts = types.ModuleType("lerobot.scripts")
        scripts.__path__ = []
        module = types.ModuleType("lerobot.scripts.lerobot_train")
        if entry is not None:
            module.train = entry
        return {"lerobot": package, "lerobot.scripts": scripts,
                "lerobot.scripts.lerobot_train": module}

    def test_calls_decorated_entry_without_bypassing_native_parser(self):
        previous = sys.argv
        seen = []

        def undecorated_train(cfg):
            raise AssertionError("Wrapper must let LeRobot parse its own TrainPipelineConfig")

        @wraps(undecorated_train)
        def decorated_train(*args, **kwargs):
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {})
            seen.append(list(sys.argv))

        restore = Mock(side_effect=lambda: self.assertIs(sys.argv, previous))
        with patch.dict(sys.modules, self.entry_modules(decorated_train)), \
                patch.object(runtime, "install_strict_native_loader", return_value=restore) as install:
            native.invoke_official(["--steps=2", "--batch_size=1"])
        self.assertEqual(seen, [["lerobot_train", "--steps=2", "--batch_size=1"]])
        self.assertIs(sys.argv, previous)
        install.assert_called_once_with(checkpoint_format="base")
        restore.assert_called_once_with()

    def test_restores_process_state_after_training_failure_and_interrupt(self):
        for failure in (RuntimeError("native training failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                previous = sys.argv

                def failing_entry():
                    self.assertEqual(sys.argv, ["lerobot_train", "--resume=true"])
                    raise failure

                restore = Mock(side_effect=lambda: self.assertIs(sys.argv, previous))
                with patch.dict(sys.modules, self.entry_modules(failing_entry)), \
                        patch.object(runtime, "install_strict_native_loader", return_value=restore) as install:
                    with self.assertRaises(type(failure)):
                        native.invoke_official(["--resume=true"])
                self.assertIs(sys.argv, previous)
                install.assert_called_once_with(checkpoint_format="native")
                restore.assert_called_once_with()

    def test_restores_strict_loader_when_official_import_fails(self):
        previous = sys.argv
        restore = Mock()
        with patch.dict(sys.modules, self.entry_modules(None)), \
                patch.object(runtime, "install_strict_native_loader", return_value=restore):
            with self.assertRaises(ImportError):
                native.invoke_official([])
        self.assertIs(sys.argv, previous)
        restore.assert_called_once_with()

    @unittest.skipIf(torch is None, "torch and safetensors are required")
    def test_resume_entry_reloads_independent_embedding_weights_without_base_remap(self):
        class CheckpointPolicy(torch.nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(3, 2)
                self.model.lm_head = torch.nn.Linear(2, 3, bias=False)

            def _fix_pytorch_state_dict_keys(self, *args):
                raise AssertionError("Resume must never run the base migration")

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise AssertionError("Resume must install the strict native loader")

        config = types.SimpleNamespace(device="cpu", dtype="float32")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text('{"type":"pi05"}')
            source = CheckpointPolicy(config)
            with torch.no_grad():
                source.model.embed_tokens.weight.fill_(1.)
                source.model.lm_head.weight.fill_(7.)
            expected = source.state_dict()
            save_file(expected, str(directory / "model.safetensors"))
            loaded = []

            def official_entry():
                self.assertEqual(sys.argv, ["lerobot_train", "--resume=true"])
                policy = CheckpointPolicy.from_pretrained(directory, config=config)
                self.assertFalse(policy._fr3_weight_load_report["key_remap_applied"])
                for name, tensor in policy.state_dict().items():
                    self.assertTrue(torch.equal(tensor, expected[name]), name)
                loaded.append(policy)

            modules = self.entry_modules(official_entry)
            for name in ("lerobot.policies", "lerobot.policies.pi05"):
                package = types.ModuleType(name)
                package.__path__ = []
                modules[name] = package
            module = types.ModuleType("lerobot.policies.pi05.modeling_pi05")
            module.PI05Policy = CheckpointPolicy
            modules[module.__name__] = module
            original = CheckpointPolicy.__dict__["from_pretrained"]
            with patch.dict(sys.modules, modules):
                native.invoke_official(["--resume=true"])
            self.assertEqual(len(loaded), 1)
            self.assertIs(CheckpointPolicy.__dict__["from_pretrained"], original)


class NativeBootstrapTests(unittest.TestCase):
    def test_fresh_launch_prepares_only_adapter_then_delegates_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = assets(Path(tmp))
            output = Path(config["output_dir"])
            seen = {}

            def prepare(base, destination, *, stats, tokenizer_source, policy_overrides):
                destination = Path(destination)
                # The actual preparer owns creating this new directory.
                self.assertFalse(destination.exists(), "Bootstrap passed an existing adapter directory")
                self.assertFalse(output.exists(), "Native trainer must create its output directory")
                self.assertNotEqual(destination, output)
                self.assertEqual(Path(base), Path(config["base_model"]))
                self.assertEqual(stats["action"]["count"], [3])
                self.assertEqual(tokenizer_source, config["tokenizer"])
                self.assertFalse(policy_overrides["gradient_checkpointing"])
                destination.mkdir()
                seen["prepared"] = destination
                return {"large_model_allocated": False}

            def invoke(flags):
                self.assertFalse(output.exists(), "Wrapper precreated the native output directory")
                values = dict(flag[2:].split("=", 1) for flag in flags)
                self.assertEqual(Path(values["policy.path"]), seen["prepared"])
                self.assertEqual(values["dataset.root"], config["dataset"])
                self.assertEqual(values["dataset.eval_split"], "0.0")
                self.assertEqual(values["eval_steps"], "0")
                self.assertEqual(values["batch_size"], "16")
                self.assertNotIn("config_path", values)
                seen["invoked"] = True

            original_import = builtins.__import__

            def no_training_imports(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "transformers", "lerobot", "accelerate"}:
                    raise AssertionError(f"Wrapper imported a training component: {name}")
                return original_import(name, *args, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(patch.object(data, "inspect_contract", return_value={
                    "num_frames": 3, "num_episodes": 1, "fps": 15}))
                stack.enter_context(patch.object(data, "stats_to_tensors", side_effect=lambda value: value))
                stack.enter_context(patch.object(runtime, "check_runtime", return_value=native_runtime()))
                prepare_spy = stack.enter_context(patch.object(runtime, "prepare_native_policy", side_effect=prepare))
                invoke_spy = stack.enter_context(patch.object(native, "invoke_official", side_effect=invoke))
                stack.enter_context(patch("builtins.__import__", side_effect=no_training_imports))
                stack.enter_context(redirect_stdout(io.StringIO()))
                native.launch(config)
            self.assertTrue(seen["invoked"])
            prepare_spy.assert_called_once()
            invoke_spy.assert_called_once()
            self.assertFalse(output.exists())

    def test_existing_run_is_rejected_before_data_or_runtime_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = assets(Path(tmp))
            Path(config["output_dir"]).mkdir()
            with patch.object(data, "inspect_contract") as inspect, \
                    patch.object(runtime, "check_runtime") as check, \
                    patch.object(native, "invoke_official") as invoke:
                with self.assertRaises(FileExistsError):
                    native.launch(config)
            inspect.assert_not_called()
            check.assert_not_called()
            invoke.assert_not_called()

    def test_incomplete_statistics_abort_before_preparation_or_training(self):
        for feature in ("observation.state", "action"):
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                config = assets(Path(tmp))
                stats = statistics()
                stats[feature]["count"] = [2]
                (Path(config["dataset"]) / "meta/stats.json").write_text(json.dumps(stats))
                with patch.object(data, "inspect_contract", return_value={"num_frames": 3}), \
                        patch.object(runtime, "check_runtime") as check, \
                        patch.object(runtime, "prepare_native_policy") as prepare, \
                        patch.object(native, "invoke_official") as invoke:
                    with self.assertRaisesRegex(ValueError, "statistics must cover all 3 frames"):
                        native.launch(config)
                check.assert_not_called()
                prepare.assert_not_called()
                invoke.assert_not_called()
                self.assertFalse(Path(config["output_dir"]).exists())

    def test_legacy_runtime_is_rejected_without_writing_or_starting_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = assets(Path(tmp))
            with patch.object(data, "inspect_contract", return_value={"num_frames": 3}), \
                    patch.object(runtime, "check_runtime", return_value=native_runtime("0.4.4")), \
                    patch.object(runtime, "prepare_native_policy") as prepare, \
                    patch.object(native, "invoke_official") as invoke:
                with self.assertRaisesRegex(RuntimeError, "LeRobot 0.6.1"):
                    native.launch(config)
            prepare.assert_not_called()
            invoke.assert_not_called()
            self.assertFalse(Path(config["output_dir"]).exists())

    def test_native_resume_skips_base_resolution_and_policy_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = assets(Path(tmp))
            config.update(resume=str(Path(tmp) / "checkpoint"), config_path=str(Path(tmp) / "checkpoint/pretrained_model/train_config.json"))
            Path(config["output_dir"]).mkdir()
            with patch.object(data, "inspect_contract", return_value={"num_frames": 3, "num_episodes": 1, "fps": 15}), \
                    patch.object(runtime, "check_runtime", return_value=native_runtime()), \
                    patch.object(native, "resolve_local_base") as resolve, \
                    patch.object(runtime, "prepare_native_policy") as prepare, \
                    patch.object(native, "invoke_official") as invoke, redirect_stdout(io.StringIO()):
                native.launch(config)
            resolve.assert_not_called()
            prepare.assert_not_called()
            flags = invoke.call_args.args[0]
            self.assertIn("--resume=true", flags)
            self.assertIn("--config_path=" + config["config_path"], flags)
            self.assertFalse(any(flag.startswith("--policy.path=") for flag in flags))


class CachedBaseResolutionTests(unittest.TestCase):
    def test_existing_directory_never_contacts_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = Mock(side_effect=AssertionError("Must use existing local directory"))
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                self.assertEqual(native.resolve_local_base(tmp), Path(tmp).resolve())
            hub.snapshot_download.assert_not_called()

    def test_cached_pi05_base_is_resolved_with_local_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = types.ModuleType("huggingface_hub")
            hub.snapshot_download = Mock(return_value=tmp)
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                self.assertEqual(native.resolve_local_base("lerobot/pi05_base"), Path(tmp).resolve())
            hub.snapshot_download.assert_called_once_with(repo_id="lerobot/pi05_base", local_files_only=True)

    def test_missing_cache_does_not_retry_online_or_accept_other_model_ids(self):
        hub = types.ModuleType("huggingface_hub")
        missing = FileNotFoundError("No cached snapshot")
        hub.snapshot_download = Mock(side_effect=missing)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            with self.assertRaisesRegex(RuntimeError, "not in the selected Hugging Face cache") as caught:
                native.resolve_local_base("lerobot/pi05_base")
            self.assertIs(caught.exception.__cause__, missing)
            with self.assertRaises(FileNotFoundError):
                native.resolve_local_base("other/missing_model")
        hub.snapshot_download.assert_called_once_with(repo_id="lerobot/pi05_base", local_files_only=True)


if __name__ == "__main__":
    unittest.main()
