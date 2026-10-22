"""CPU integration checks. Set FR3_TEST_DATASET to test another dataset copy."""

import json
import os
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fr3_pi05.data import (
    ACTION_NAMES, CAMERA_KEYS, DatasetContractError, STATE_NAMES,
    compute_training_stats, inspect_contract, make_dataset, stats_to_json, stats_to_tensors,
)


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data/lerobot/fr3_avocado_v30"
DATASET_ROOT = Path(os.environ.get("FR3_TEST_DATASET", DEFAULT_DATASET))


class JsonPreflightTest(unittest.TestCase):
    def test_bad_version_fails_before_importing_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            meta = Path(directory) / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
            (meta / "conversion_manifest.json").write_text("{}")
            with patch("fr3_pi05.data._parquet_module", side_effect=AssertionError("must not import")):
                with self.assertRaisesRegex(DatasetContractError, "v3.0"):
                    inspect_contract(directory)

    def test_json_rejects_nonfinite(self):
        with self.assertRaises(DatasetContractError):
            stats_to_json({"action": {"mean": [float("nan")]}})


@unittest.skipUnless((DATASET_ROOT / "meta/info.json").is_file(), "local FR3 dataset not supplied")
class CurrentDatasetIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = inspect_contract(DATASET_ROOT)
        cls.dataset = make_dataset(DATASET_ROOT, chunk_size=50)
        cls.stats = compute_training_stats(cls.dataset)

    def test_full_contract_and_every_episode_selected(self):
        self.assertEqual(self.contract["feature_order"]["observation.state"], STATE_NAMES)
        self.assertEqual(self.contract["feature_order"]["action"], ACTION_NAMES)
        self.assertIsNone(self.dataset.episodes)
        self.assertEqual(len(self.dataset), self.contract["num_frames"])
        self.assertEqual(len(self.contract["training_episodes"]), self.contract["num_episodes"])

    def test_rejects_wrong_units_order_and_control_mode_without_data_changes(self):
        from fr3_pi05 import data

        real_reader = data._read_json
        for change in ("units", "joint_order", "gripper_mode"):
            def changed_reader(path):
                value = copy.deepcopy(real_reader(path))
                if path.name == "conversion_manifest.json":
                    if change == "units":
                        value["feature_units"][0] = "degree"
                    elif change == "gripper_mode":
                        value["gripper_control_profiles"][0]["control_policy"] = "grasp_hold_v1"
                elif path.name == "info.json" and change == "joint_order":
                    names = value["features"]["action"]["names"]
                    names[0], names[1] = names[1], names[0]
                return value
            with self.subTest(change=change), patch("fr3_pi05.data._read_json", side_effect=changed_reader):
                with self.assertRaises(DatasetContractError):
                    inspect_contract(DATASET_ROOT)

    def test_actual_cpu_video_sample_and_action_chunk(self):
        import torch

        sample = self.dataset[0]
        self.assertEqual(tuple(sample["observation.state"].shape), (8,))
        self.assertEqual(tuple(sample["action"].shape), (50, 8))
        for key in CAMERA_KEYS:
            self.assertEqual(tuple(sample[key].shape), (3, 480, 640))
            self.assertTrue(torch.isfinite(sample[key]).all())
            self.assertGreaterEqual(float(sample[key].min()), 0.0)
            self.assertLessEqual(float(sample[key].max()), 1.0)

    def test_actions_repeat_at_episode_end_never_cross_boundary(self):
        import torch

        # Test an interior boundary and the final dataset row: both use the
        # official reader rather than reproducing its clipping implementation.
        for episode in (self.dataset.meta.episodes[0], self.dataset.meta.episodes[self.contract["num_episodes"] - 1]):
            index = int(episode["dataset_to_index"]) - 1
            sample = self.dataset[index]
            expected = torch.as_tensor(self.dataset.hf_dataset[index]["action"])
            self.assertTrue(torch.equal(sample["action"], expected.expand(50, -1)))
            self.assertFalse(bool(sample["action_is_pad"][0]))
            self.assertTrue(bool(sample["action_is_pad"][1:].all()))

    def test_exported_stats_match_all_cleaned_scalar_rows(self):
        import numpy as np

        for key in ("action", "observation.state"):
            values = np.asarray(self.dataset.hf_dataset.with_format("numpy")[key], dtype=np.float64)
            self.assertEqual(self.stats[key]["count"], [self.contract["num_frames"]])
            for name, expected in (("mean", values.mean(axis=0)), ("std", values.std(axis=0)),
                                   ("min", values.min(axis=0)), ("max", values.max(axis=0))):
                np.testing.assert_allclose(self.stats[key][name], expected, rtol=1e-12, atol=1e-12)
            for name, quantile in (("q01", .01), ("q10", .1), ("q50", .5), ("q90", .9), ("q99", .99)):
                np.testing.assert_allclose(self.stats[key][name], np.quantile(values, quantile, axis=0))
        original = json.loads((DATASET_ROOT / "meta/stats.json").read_text())
        for key in ("action", "observation.state"):
            self.assertEqual(original[key]["count"], [self.contract["num_frames"]])
            for name in ("mean", "std", "min", "max", "q01", "q10", "q50", "q90", "q99"):
                np.testing.assert_allclose(original[key][name], self.stats[key][name], rtol=1e-12, atol=1e-12)
        for key in CAMERA_KEYS:
            self.assertEqual(self.stats[key], original[key])
        json.dumps(self.stats, allow_nan=False)

    def test_tensor_processor_stats_round_trip(self):
        import torch

        tensors = stats_to_tensors(self.stats)
        self.assertEqual(tuple(tensors["action"]["q99"].shape), (8,))
        self.assertEqual(tensors["action"]["q99"].dtype, torch.float32)
        self.assertEqual(set(stats_to_json(tensors)), set(self.stats))


if __name__ == "__main__":
    unittest.main()
