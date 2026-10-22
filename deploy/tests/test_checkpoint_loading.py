"""Native export regression tests using tiny CPU tensors, never PI05/GPU/ROS."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
    from torch import nn
    from safetensors.torch import save_file, save_model
except ImportError:
    torch = nn = save_file = save_model = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_loading import load_native_weights_strict


class TinyPolicy(nn.Module if nn is not None else object):
    def __init__(self, tied=False):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(3, 2)
        self.model.lm_head = nn.Linear(2, 3, bias=False)
        if tied:
            self.model.lm_head.weight = self.model.embed_tokens.weight
        self.register_buffer("counter", torch.tensor([0], dtype=torch.int64))

    def _fix_pytorch_state_dict_keys(self, *args, **kwargs):
        raise AssertionError("Native checkpoint must never invoke base-model key migration")


@unittest.skipUnless(torch is not None and save_file is not None, "requires existing torch + safetensors environment")
class NativeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, policy, **overrides):
        # clone each entry so save_file can represent separate tied aliases.
        state = {name: value.clone() for name, value in policy.state_dict().items()}
        state.update(overrides)
        save_file(state, str(self.path))
        return state

    def test_distinct_native_lm_head_and_embed_values_are_preserved(self):
        source = TinyPolicy()
        state = self.write(source, **{
            "model.embed_tokens.weight": torch.full((3, 2), 1.),
            "model.lm_head.weight": torch.full((3, 2), 7.),
        })
        target = TinyPolicy()
        report = load_native_weights_strict(target, self.path)
        self.assertTrue(torch.equal(target.model.embed_tokens.weight, state["model.embed_tokens.weight"]))
        self.assertTrue(torch.equal(target.model.lm_head.weight, state["model.lm_head.weight"]))
        self.assertFalse(report["key_remap_applied"])
        self.assertEqual(report["shared_aliases_filled"], [])
        self.assertTrue(report["strict"])

    def test_conflicting_true_tied_weights_fail_before_any_mutation(self):
        target = TinyPolicy(tied=True)
        before = target.model.embed_tokens.weight.detach().clone()
        self.write(target, **{
            "model.embed_tokens.weight": torch.full((3, 2), 1.),
            "model.lm_head.weight": torch.full((3, 2), 2.),
        })
        with self.assertRaisesRegex(RuntimeError, "Conflicting.*tied"):
            load_native_weights_strict(target, self.path)
        self.assertTrue(torch.equal(target.model.embed_tokens.weight, before))

    def test_missing_shared_alias_from_real_save_model_is_completed(self):
        source = TinyPolicy(tied=True)
        with torch.no_grad():
            source.model.embed_tokens.weight.fill_(3.)
        save_model(source, str(self.path))
        target = TinyPolicy(tied=True)
        report = load_native_weights_strict(target, self.path)
        self.assertEqual(len(report["shared_aliases_filled"]), 1)
        self.assertEqual(report["loaded_tensor_count"], report["source_tensor_count"] + 1)
        self.assertTrue(torch.equal(target.model.lm_head.weight, source.model.lm_head.weight))
        self.assertIs(target.model.embed_tokens.weight, target.model.lm_head.weight)

    def test_missing_independent_weight_is_not_filled_from_lm_head(self):
        target = TinyPolicy()
        state = self.write(target)
        del state["model.embed_tokens.weight"]
        save_file(state, str(self.path))
        with self.assertRaisesRegex(RuntimeError, "missing independent.*model.embed_tokens.weight"):
            load_native_weights_strict(target, self.path)

    def test_equal_explicit_tied_aliases_are_allowed(self):
        target = TinyPolicy(tied=True)
        self.write(target)
        report = load_native_weights_strict(target, self.path)
        self.assertEqual(report["shared_aliases_filled"], [])

    def test_missing_entire_shared_group_fails(self):
        target = TinyPolicy(tied=True)
        save_file({"counter": torch.tensor([4])}, str(self.path))
        with self.assertRaisesRegex(RuntimeError, "missing independent"):
            load_native_weights_strict(target, self.path)

    def test_shared_storage_of_distinct_parameters_does_not_allow_alias_completion(self):
        target = TinyPolicy()
        target.model.lm_head.weight = nn.Parameter(target.model.embed_tokens.weight.detach())
        self.assertIsNot(target.model.lm_head.weight, target.model.embed_tokens.weight)
        state = self.write(target)
        del state["model.lm_head.weight"]
        save_file(state, str(self.path))
        with self.assertRaisesRegex(RuntimeError, "missing independent.*model.lm_head.weight"):
            load_native_weights_strict(target, self.path)

    def test_shared_persistent_buffer_alias_is_completed_but_nonpersistent_is_ignored(self):
        target = TinyPolicy()
        target.register_buffer("counter_alias", target.counter)
        target.register_buffer("scratch_alias", target.counter, persistent=False)
        state = self.write(target)
        del state["counter_alias"]
        state["counter"] = torch.tensor([8])
        save_file(state, str(self.path))
        report = load_native_weights_strict(target, self.path)
        self.assertEqual(report["shared_aliases_filled"], ["counter_alias"])
        self.assertEqual(target.counter_alias.item(), 8)

    def test_unexpected_key_is_rejected(self):
        target = TinyPolicy()
        self.write(target, accidental_extra_layer=torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, "unexpected keys.*accidental_extra_layer"):
            load_native_weights_strict(target, self.path)

    def test_removed_model_prefix_is_rejected_not_migrated(self):
        target = TinyPolicy()
        state = self.write(target)
        state["lm_head.weight"] = state.pop("model.lm_head.weight")
        save_file(state, str(self.path))
        with self.assertRaisesRegex(RuntimeError, "no prefix/key migration"):
            load_native_weights_strict(target, self.path)

    def test_shape_mismatch_fails_before_mutation(self):
        target = TinyPolicy()
        before = {name: value.clone() for name, value in target.state_dict().items()}
        self.write(target, **{"model.lm_head.weight": torch.zeros(4, 2), "counter": torch.tensor([9])})
        with self.assertRaisesRegex(RuntimeError, "shape/type mismatch"):
            load_native_weights_strict(target, self.path)
        for name, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


if __name__ == "__main__":
    unittest.main()
