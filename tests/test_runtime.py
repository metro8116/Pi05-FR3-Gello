"""Small offline tests: no π0.5 allocation, network, GPU, or robot commands."""

from __future__ import annotations

import json
import importlib.metadata
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fr3_pi05 import runtime

try:
    import torch
    from safetensors.torch import save_file, save_model
except ImportError:
    torch = None


if torch is not None:

    class TinyPolicy(torch.nn.Module):
        def __init__(self, tied=False):
            super().__init__()
            self.config = types.SimpleNamespace()
            self.model = torch.nn.Module()
            self.model.first = torch.nn.Linear(2, 2)
            self.model.second = torch.nn.Linear(2, 2)
            if tied:
                self.model.second.weight = self.model.first.weight

        def _fix_pytorch_state_dict_keys(self, state, config):
            return state


    class TinyPi05Weights(torch.nn.Module):
        """Real PI05 key layout/remapper, with only two tiny CPU parameters."""
        EMBED = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        HEAD = "model.paligemma_with_expert.paligemma.lm_head.weight"

        def __init__(self, tied=False):
            super().__init__()
            self.config = types.SimpleNamespace()
            self.model = torch.nn.Module()
            self.model.paligemma_with_expert = torch.nn.Module()
            paligemma = self.model.paligemma_with_expert.paligemma = torch.nn.Module()
            paligemma.model = torch.nn.Module()
            paligemma.model.language_model = torch.nn.Module()
            paligemma.model.language_model.embed_tokens = torch.nn.Embedding(3, 2)
            paligemma.lm_head = torch.nn.Linear(2, 3, bias=False)
            if tied:
                paligemma.lm_head.weight = paligemma.model.language_model.embed_tokens.weight
            self.register_buffer("counter", torch.tensor([0], dtype=torch.int64))

        @property
        def embed(self):
            return self.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens

        @property
        def head(self):
            return self.model.paligemma_with_expert.paligemma.lm_head

        def _fix_pytorch_state_dict_keys(self, state, config):
            # Import the official method, without constructing a full policy.
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            return PI05Policy._fix_pytorch_state_dict_keys(self, state, config)


@unittest.skipIf(torch is None, "torch and safetensors are required for tiny strict-loader tests")
class StrictWeightsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "model.safetensors"

    def tearDown(self):
        self.directory.cleanup()

    def test_saved_checkpoint_reloads_exactly(self):
        source, target = TinyPolicy(), TinyPolicy()
        save_model(source, str(self.path))
        report = runtime.load_weights_strict(target, self.path)
        self.assertTrue(report["strict"])
        self.assertEqual(report["missing_keys"], [])
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[name]))

    def test_unprefixed_base_keys_are_loaded(self):
        source, target = TinyPolicy(), TinyPolicy()
        save_file({name.removeprefix("model."): value for name, value in source.state_dict().items()}, str(self.path))
        report = runtime.load_weights_strict(target, self.path)
        self.assertTrue(report["key_remap_applied"])
        self.assertTrue(torch.equal(source.model.first.weight, target.model.first.weight))

    def test_tied_alias_removed_by_safetensors_is_restored(self):
        source, target = TinyPolicy(tied=True), TinyPolicy(tied=True)
        save_model(source, str(self.path))
        report = runtime.load_weights_strict(target, self.path)
        self.assertEqual(len(report["shared_aliases_filled"]), 1)
        self.assertTrue(torch.equal(source.model.second.weight, target.model.second.weight))

    def test_missing_independent_tensor_aborts(self):
        source = TinyPolicy()
        state = dict(source.state_dict())
        state.pop("model.first.bias")
        save_file(state, str(self.path))
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            runtime.load_weights_strict(TinyPolicy(), self.path)

    def test_shape_mismatch_aborts(self):
        state = dict(TinyPolicy().state_dict())
        state["model.first.bias"] = torch.zeros(9)
        save_file(state, str(self.path))
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            runtime.load_weights_strict(TinyPolicy(), self.path)

    def test_conflicting_tied_values_abort(self):
        state = {name: value.clone() for name, value in TinyPolicy(tied=True).state_dict().items()}
        state["model.second.weight"] += 1
        save_file(state, str(self.path))
        with self.assertRaisesRegex(RuntimeError, "Conflicting values for tied"):
            runtime.load_weights_strict(TinyPolicy(tied=True), self.path)

    def test_prefix_collision_aborts(self):
        save_file({"model.first.bias": torch.zeros(2), "first.bias": torch.zeros(2)}, str(self.path))
        with self.assertRaisesRegex(RuntimeError, "prefix collision"):
            runtime.load_weights_strict(TinyPolicy(), self.path)

    def test_nonexistent_weights_never_return_random_policy(self):
        with self.assertRaises(FileNotFoundError):
            runtime.load_weights_strict(TinyPolicy(), self.path)


@unittest.skipIf(torch is None, "torch and safetensors are required for native checkpoint tests")
class NativeWeightsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "model.safetensors"

    def tearDown(self):
        self.directory.cleanup()

    def write(self, target, **overrides):
        state = {name: value.clone() for name, value in target.state_dict().items()}
        state.update(overrides)
        save_file(state, str(self.path))
        return state

    def before(self, target):
        return {name: value.clone() for name, value in target.state_dict().items()}

    def unchanged(self, target, before):
        for name, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def require_official_061_remapper(self):
        try:
            version = importlib.metadata.version("lerobot")
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            self.skipTest(str(exc))
        if version != "0.6.1":
            self.skipTest(f"Requires the actual LeRobot 0.6.1 PI05 remapper; found legacy {version}")

    def test_real_pi05_base_remapper_reproduces_native_diverged_weight_collision(self):
        self.require_official_061_remapper()
        target = TinyPi05Weights()
        # Native PI05's model parameters are model-prefixed. The root buffer is
        # only a native-schema test fixture, not an input to base migration.
        del target.counter
        self.write(target, **{TinyPi05Weights.EMBED: torch.full((3, 2), 1.),
                              TinyPi05Weights.HEAD: torch.full((3, 2), 7.)})
        before = self.before(target)
        with self.assertRaisesRegex(RuntimeError, "Conflicting remapped checkpoint key: .*embed_tokens.weight"):
            runtime.load_weights_strict(target, self.path)
        self.unchanged(target, before)
        # The same complete checkpoint is valid when loaded as native names.
        with patch.object(target, "_fix_pytorch_state_dict_keys", side_effect=AssertionError("no native migration")):
            report = runtime.load_native_weights_strict(target, self.path)
        self.assertTrue(torch.equal(target.embed.weight, torch.full((3, 2), 1.)))
        self.assertTrue(torch.equal(target.head.weight, torch.full((3, 2), 7.)))
        self.assertFalse(report["key_remap_applied"])
        self.assertEqual(report["format"], "native_state_dict")

    def test_base_migration_still_maps_unprefixed_lm_head_to_missing_embed(self):
        self.require_official_061_remapper()
        target = TinyPi05Weights()
        del target.counter
        # Initial base assets can omit embed_tokens and need the official alias.
        save_file({TinyPi05Weights.HEAD.removeprefix("model."): torch.full((3, 2), 4.)}, str(self.path))
        report = runtime.load_weights_strict(target, self.path)
        self.assertEqual(report["format"], "base_migration")
        self.assertTrue(torch.equal(target.embed.weight, torch.full((3, 2), 4.)))
        self.assertTrue(torch.equal(target.head.weight, torch.full((3, 2), 4.)))

    def test_native_independent_embed_and_lm_head_values_are_preserved_without_remapper(self):
        target = TinyPi05Weights()
        self.write(target, **{TinyPi05Weights.EMBED: torch.full((3, 2), 1.),
                              TinyPi05Weights.HEAD: torch.full((3, 2), 7.)})
        with patch.object(target, "_fix_pytorch_state_dict_keys", side_effect=AssertionError("must never migrate")):
            report = runtime.load_native_weights_strict(target, self.path)
        self.assertTrue(torch.equal(target.embed.weight, torch.full((3, 2), 1.)))
        self.assertTrue(torch.equal(target.head.weight, torch.full((3, 2), 7.)))
        self.assertFalse(report["key_remap_applied"])
        self.assertEqual(report["shared_aliases_filled"], [])

    def test_native_missing_independent_embed_is_not_filled_from_lm_head(self):
        target = TinyPi05Weights()
        state = self.write(target)
        del state[TinyPi05Weights.EMBED]
        save_file(state, str(self.path))
        before = self.before(target)
        with self.assertRaisesRegex(RuntimeError, "missing independent.*embed_tokens.weight"):
            runtime.load_native_weights_strict(target, self.path)
        self.unchanged(target, before)

    def test_native_same_storage_distinct_parameters_are_not_treated_as_tied(self):
        target = TinyPi05Weights()
        target.head.weight = torch.nn.Parameter(target.embed.weight.detach())
        self.assertIsNot(target.embed.weight, target.head.weight)
        state = self.write(target)
        del state[TinyPi05Weights.HEAD]
        save_file(state, str(self.path))
        before = self.before(target)
        with self.assertRaisesRegex(RuntimeError, "missing independent.*lm_head.weight"):
            runtime.load_native_weights_strict(target, self.path)
        self.unchanged(target, before)

    def test_native_save_model_true_shared_alias_is_completed(self):
        source, target = TinyPi05Weights(tied=True), TinyPi05Weights(tied=True)
        with torch.no_grad():
            source.embed.weight.fill_(3.)
        save_model(source, str(self.path))
        report = runtime.load_native_weights_strict(target, self.path)
        self.assertEqual(len(report["shared_aliases_filled"]), 1)
        self.assertIs(target.head.weight, target.embed.weight)
        self.assertTrue(torch.equal(target.embed.weight, source.embed.weight))

    def test_native_conflicting_true_aliases_fail_before_copy(self):
        target = TinyPi05Weights(tied=True)
        self.write(target, **{TinyPi05Weights.EMBED: torch.full((3, 2), 1.),
                              TinyPi05Weights.HEAD: torch.full((3, 2), 7.)})
        before = self.before(target)
        with self.assertRaisesRegex(RuntimeError, "Conflicting.*tied native"):
            runtime.load_native_weights_strict(target, self.path)
        self.unchanged(target, before)

    def test_native_persistent_buffer_alias_is_completed_but_scratch_is_ignored(self):
        target = TinyPi05Weights()
        target.register_buffer("counter_alias", target.counter)
        target.register_buffer("scratch_alias", target.counter, persistent=False)
        state = self.write(target)
        del state["counter_alias"]
        state["counter"] = torch.tensor([8])
        save_file(state, str(self.path))
        report = runtime.load_native_weights_strict(target, self.path)
        self.assertEqual(report["shared_aliases_filled"], ["counter_alias"])
        self.assertEqual(target.counter_alias.item(), 8)

    def test_native_unexpected_key_and_removed_prefix_fail_before_copy(self):
        for mode in ("unexpected", "removed_prefix"):
            with self.subTest(mode=mode):
                target = TinyPi05Weights()
                state = self.write(target, counter=torch.tensor([9]))
                if mode == "unexpected":
                    state["accidental_extra"] = torch.ones(2)
                else:
                    state[TinyPi05Weights.HEAD.removeprefix("model.")] = state.pop(TinyPi05Weights.HEAD)
                save_file(state, str(self.path))
                before = self.before(target)
                with self.assertRaisesRegex(RuntimeError, "unexpected keys; no prefix/key migration"):
                    runtime.load_native_weights_strict(target, self.path)
                self.unchanged(target, before)

    def test_native_invalid_shape_fails_before_any_copy(self):
        target = TinyPi05Weights()
        self.write(target, counter=torch.tensor([9]), **{TinyPi05Weights.HEAD: torch.zeros(4, 2)})
        before = self.before(target)
        with self.assertRaisesRegex(RuntimeError, "shape/type mismatch"):
            runtime.load_native_weights_strict(target, self.path)
        self.unchanged(target, before)

    def test_native_missing_entire_shared_group_and_empty_file_fail(self):
        for state in ({"counter": torch.tensor([9])}, {}):
            with self.subTest(keys=list(state)):
                target = TinyPi05Weights(tied=True)
                save_file(state, str(self.path))
                before = self.before(target)
                with self.assertRaises(RuntimeError):
                    runtime.load_native_weights_strict(target, self.path)
                self.unchanged(target, before)


@unittest.skipIf(torch is None, "torch and safetensors are required for loader patch tests")
class NativeLoaderPatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        (self.base / "config.json").write_text('{"type":"pi05"}')
        save_model(TinyPolicy(), str(self.base / "model.safetensors"))

    def tearDown(self):
        self.directory.cleanup()

    def fake_policy(self):
        class FakeNativePolicy(TinyPolicy):
            migrations = 0

            def __init__(self, config, **kwargs):
                super().__init__()
                self.config, self.received_kwargs = config, kwargs
                self.model.first.weight.requires_grad_(not config.freeze_vision_encoder)

            def _fix_pytorch_state_dict_keys(self, state, config):
                type(self).migrations += 1
                return state

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise AssertionError("unsafe original loader must not be called")

        module = types.ModuleType("lerobot.policies.pi05.modeling_pi05")
        module.PI05Policy = FakeNativePolicy
        return module, FakeNativePolicy

    def test_explicit_base_and_native_dispatch_keep_config_freezing_kwargs_and_restore(self):
        for mode in ("base", "native"):
            with self.subTest(mode=mode):
                module, policy_class = self.fake_policy()
                original = policy_class.__dict__["from_pretrained"]
                config = types.SimpleNamespace(freeze_vision_encoder=True, device="cpu", dtype="float32")
                with patch.dict(sys.modules, {"lerobot.policies.pi05.modeling_pi05": module}):
                    restore = runtime.install_strict_native_loader(checkpoint_format=mode)
                    try:
                        loaded = policy_class.from_pretrained(self.base, config=config, dataset_meta="native metadata")
                        self.assertIs(loaded.config, config)
                        self.assertFalse(loaded.model.first.weight.requires_grad)
                        self.assertEqual(loaded.received_kwargs["dataset_meta"], "native metadata")
                        self.assertEqual(policy_class.migrations, 1 if mode == "base" else 0)
                        self.assertEqual(loaded._fr3_weight_load_report["format"],
                                         "base_migration" if mode == "base" else "native_state_dict")
                    finally:
                        restore()
                self.assertIs(policy_class.__dict__["from_pretrained"], original)
                self.assertNotIn("_fr3_strict_native_loader", policy_class.__dict__)
                self.assertNotIn("_fr3_strict_checkpoint_format", policy_class.__dict__)

    def test_native_patch_never_falls_back_and_can_be_restored_after_failure(self):
        module, policy_class = self.fake_policy()
        original = policy_class.__dict__["from_pretrained"]
        state = dict(TinyPolicy().state_dict())
        del state["model.first.bias"]
        save_file(state, str(self.base / "model.safetensors"))
        with patch.dict(sys.modules, {"lerobot.policies.pi05.modeling_pi05": module}):
            restore = runtime.install_strict_native_loader(checkpoint_format="native")
            try:
                with self.assertRaisesRegex(RuntimeError, "missing independent"):
                    policy_class.from_pretrained(self.base, config=types.SimpleNamespace(freeze_vision_encoder=False))
            finally:
                restore()
        self.assertEqual(policy_class.migrations, 0)
        self.assertIs(policy_class.__dict__["from_pretrained"], original)

    def test_different_nested_format_is_rejected_instead_of_reusing_wrong_loader(self):
        module, policy_class = self.fake_policy()
        original = policy_class.__dict__["from_pretrained"]
        with patch.dict(sys.modules, {"lerobot.policies.pi05.modeling_pi05": module}):
            restore = runtime.install_strict_native_loader(checkpoint_format="base")
            try:
                redundant_restore = runtime.install_strict_native_loader(checkpoint_format="base")
                redundant_restore()
                self.assertIsNot(policy_class.__dict__["from_pretrained"], original)
                with self.assertRaisesRegex(RuntimeError, "different checkpoint format"):
                    runtime.install_strict_native_loader(checkpoint_format="native")
            finally:
                restore()
        self.assertIs(policy_class.__dict__["from_pretrained"], original)

    def test_unknown_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkpoint_format"):
            runtime.install_strict_native_loader(checkpoint_format="automatic")


class LocalAssetsTests(unittest.TestCase):
    def test_model_directory_requires_actual_pi05_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text('{"type":"act"}')
            (directory / "model.safetensors").write_bytes(b"test-only")
            with self.assertRaisesRegex(ValueError, "Expected a pi05"):
                runtime._local_model_directory(directory)


@unittest.skipIf(torch is None, "torch and safetensors are required for native adapter tests")
class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        try:
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from transformers import PreTrainedTokenizerFast
        except ImportError as exc:
            self.skipTest(str(exc))
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text(json.dumps({"type": "pi05", "device": "cpu", "dtype": "float32"}))
        save_model(TinyPolicy(), str(self.base / "model.safetensors"))
        self.tokenizer_dir = self.root / "local_tokenizer"
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
            pad_token="[PAD]", unk_token="[UNK]",
        )
        tokenizer.save_pretrained(self.tokenizer_dir)
        self.stats = {key: {"q01": [0.] * 8, "q99": [1.] * 8, "count": [12]}
                      for key in ("observation.state", "action")}

    def tearDown(self):
        if hasattr(self, "temporary"):
            self.temporary.cleanup()

    def test_prepare_uses_official_factory_no_model_allocation_or_weight_copy(self):
        from lerobot.policies.pi05 import processor_pi05

        original_constructor = processor_pi05.TokenizerProcessorStep
        original_base = {path.name: path.read_bytes() for path in self.base.iterdir()}
        prepared = self.root / "prepared"
        with patch("lerobot.policies.pi05.modeling_pi05.PI05Policy.__init__",
                   side_effect=AssertionError("must not allocate the full network")):
            report = runtime.prepare_native_policy(
                self.base, prepared, stats=self.stats, tokenizer_source=self.tokenizer_dir,
                policy_overrides={"device": "cpu", "dtype": "float32", "freeze_vision_encoder": True,
                                  "gradient_checkpointing": False, "train_expert_only": False},
            )
        self.assertIs(processor_pi05.TokenizerProcessorStep, original_constructor)
        self.assertTrue((prepared / "model.safetensors").is_symlink())
        self.assertTrue((prepared / "model.safetensors").samefile(self.base / "model.safetensors"))
        self.assertFalse(report["weights_duplicated"])
        self.assertFalse(report["large_model_allocated"])
        self.assertFalse(report["tokenizer"]["bundled"])
        config = json.loads((prepared / "config.json").read_text())
        self.assertEqual(config["type"], "pi05")
        self.assertEqual(config["input_features"]["observation.state"]["shape"], [8])
        self.assertEqual(config["output_features"]["action"]["shape"], [8])
        self.assertEqual(set(config["input_features"]), {"observation.state", *runtime.CAMERA_KEYS})
        self.assertEqual(config["max_state_dim"], 32)
        self.assertEqual(config["max_action_dim"], 32)
        self.assertFalse(config.get("use_relative_actions", False))
        self.assertFalse(config["gradient_checkpointing"])
        self.assertTrue(config["freeze_vision_encoder"])
        steps = json.loads((prepared / runtime.PREPROCESSOR_NAME).read_text())["steps"]
        token_steps = [step for step in steps if step.get("registry_name") == "tokenizer_processor"]
        self.assertEqual(len(token_steps), 1)
        self.assertEqual(token_steps[0]["config"]["tokenizer_name"], str(self.tokenizer_dir))
        self.assertFalse((prepared / "tokenizer").exists())
        self.assertTrue(all(path.read_bytes() == original_base[path.name] for path in self.base.iterdir()))
        # Use the same official pipeline loader called by the native factory,
        # without requiring optional simulation environments in this CPU test.
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition, transition_to_batch,
            policy_action_to_transition, transition_to_policy_action,
        )
        loaded_pre = PolicyProcessorPipeline.from_pretrained(
            prepared, config_filename=runtime.PREPROCESSOR_NAME, local_files_only=True,
            to_transition=batch_to_transition, to_output=transition_to_batch,
        )
        loaded_post = PolicyProcessorPipeline.from_pretrained(
            prepared, config_filename=runtime.POSTPROCESSOR_NAME, local_files_only=True,
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
        )
        batch = {"observation.state": torch.full((1, 8), .25),
                 "action": torch.full((1, 50, 8), .75),
                 "observation.images.camera_1": torch.zeros(1, 3, 480, 640),
                 "observation.images.camera_2": torch.zeros(1, 3, 480, 640),
                 "task": ["Put the avocado in the basket."]}
        processed = loaded_pre(batch)
        self.assertTrue(torch.equal(processed["action"], torch.full((1, 50, 8), .5)))
        self.assertEqual(tuple(processed["observation.language.tokens"].shape), (1, 200))
        self.assertTrue(torch.equal(loaded_post(torch.full((1, 8), .5)), torch.full((1, 8), .75)))

    def test_prepare_rejects_overwrite_and_interface_override(self):
        destination = self.root / "already_exists"
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            runtime.prepare_native_policy(self.base, destination, stats=self.stats,
                                          tokenizer_source=self.tokenizer_dir, policy_overrides={})
        with self.assertRaisesRegex(ValueError, "interface fields are fixed"):
            runtime.prepare_native_policy(self.base, self.root / "new", stats=self.stats,
                                          tokenizer_source=self.tokenizer_dir,
                                          policy_overrides={"max_action_dim": 99})
        self.assertFalse((self.root / "new").exists())

    def test_native_loader_preserves_supplied_config_and_freezing(self):
        class FakeNativePolicy(TinyPolicy):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config
                self.received_kwargs = kwargs
                self.model.first.weight.requires_grad_(not config.freeze_vision_encoder)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return "unsafe-original-loader"

        module = types.ModuleType("lerobot.policies.pi05.modeling_pi05")
        module.PI05Policy = FakeNativePolicy
        original = FakeNativePolicy.__dict__["from_pretrained"]
        config = types.SimpleNamespace(freeze_vision_encoder=True, device="cpu", dtype="float32")
        with patch.dict(sys.modules, {"lerobot.policies.pi05.modeling_pi05": module}):
            restore = runtime.install_strict_native_loader()
            try:
                loaded = FakeNativePolicy.from_pretrained(self.base, config=config, dataset_meta="native metadata")
                self.assertIs(loaded.config, config)
                self.assertFalse(loaded.model.first.weight.requires_grad)
                self.assertEqual(loaded.received_kwargs["dataset_meta"], "native metadata")
                self.assertTrue(loaded._fr3_weight_load_report["strict"])
            finally:
                restore()
        self.assertIs(FakeNativePolicy.__dict__["from_pretrained"], original)

    def test_native_loader_does_not_swallow_missing_weights(self):
        class FakeNativePolicy(TinyPolicy):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return "random-model-fallback"

        state = dict(TinyPolicy().state_dict())
        state.pop("model.first.bias")
        save_file(state, str(self.base / "model.safetensors"))
        module = types.ModuleType("lerobot.policies.pi05.modeling_pi05")
        module.PI05Policy = FakeNativePolicy
        with patch.dict(sys.modules, {"lerobot.policies.pi05.modeling_pi05": module}):
            restore = runtime.install_strict_native_loader()
            try:
                with self.assertRaises(RuntimeError):
                    FakeNativePolicy.from_pretrained(self.base, config=types.SimpleNamespace())
            finally:
                restore()


if __name__ == "__main__":
    unittest.main()
