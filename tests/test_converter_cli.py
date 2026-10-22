"""Synthetic schema-2 conversion checks; no cameras, robot, model or network."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "tools/convert_to_lerobot_v30.py"
try:
    import converter_fixture as fixture
except ImportError:
    fixture = None


class ConverterDefaultsTests(unittest.TestCase):
    def test_defaults_are_relative_to_repository_and_use_all_training_episodes(self):
        spec = importlib.util.spec_from_file_location("fr3_converter_tested", CONVERTER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            args = module.make_parser().parse_args([])
        finally:
            sys.modules.pop(spec.name, None)
        self.assertEqual(args.input_dir, ROOT / "data/raw")
        self.assertEqual(args.output_dir, ROOT / "data/lerobot/fr3_avocado_v30")
        self.assertEqual(args.repo_id, "local/fr3_avocado_pi05")
        self.assertEqual(args.val_ratio, 0.)


@unittest.skipIf(fixture is None, "Use the converter profile: numpy, OpenCV and h5py are required")
class ConverterSyntheticTests(unittest.TestCase):
    def test_empty_selection_is_not_a_successful_preflight(self):
        with tempfile.TemporaryDirectory(prefix="fr3-converter-empty-") as directory:
            raw = Path(directory) / "raw"
            record = raw / "record_001"
            record.mkdir(parents=True)
            (record / "metadata.json").write_text(json.dumps({
                "schema_version": 2, "status": "complete", "outcome": "failure"}))
            before = fixture.fingerprint(raw)
            output = Path(directory) / "absent-output"
            report = Path(directory) / "report.json"
            run = subprocess.run([sys.executable, str(CONVERTER), "--input-dir", str(raw),
                                  "--output-dir", str(output), "--dry-run", "--report", str(report)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
            self.assertEqual(json.loads(report.read_text())["status"], "no_eligible_episodes")
            self.assertFalse(output.exists())
            self.assertEqual(fixture.fingerprint(raw), before)

    def test_schema2_dry_run_regressions_leave_sources_and_output_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="fr3-converter-validation-") as directory:
            suite = fixture.Suite(CONVERTER, Path(directory))
            suite.run()
            self.assertGreaterEqual(suite.count, 18)

    def test_small_episode_exports_v3_and_decodes_both_cameras(self):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            self.skipTest(f"Use the converter export profile: {exc}")
        with tempfile.TemporaryDirectory(prefix="fr3-converter-export-") as directory:
            directory = Path(directory)
            raw, output = directory / "raw", directory / "dataset"
            fixture.make_episode(raw, task="Put the avocado in the basket.")
            before = fixture.fingerprint(raw)
            report = directory / "conversion_report.json"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
            result = subprocess.run([
                sys.executable, str(CONVERTER), "--input-dir", str(raw),
                "--output-dir", str(output), "--repo-id", "local/synthetic_fr3",
                "--report", str(report),
            ], env=env, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(fixture.fingerprint(raw), before)
            self.assertEqual(json.loads(report.read_text())["status"], "complete")
            info = json.loads((output / "meta/info.json").read_text())
            self.assertEqual(info["codebase_version"], "v3.0")
            self.assertEqual(info["total_episodes"], 1)
            self.assertEqual(info["total_frames"], fixture.N)
            self.assertEqual(info["fps"], 15)
            stats = json.loads((output / "meta/stats.json").read_text())
            for key in ("observation.state", "action"):
                self.assertEqual(info["features"][key]["shape"], [8])
                self.assertEqual(stats[key]["count"], [fixture.N])
                self.assertEqual(len(stats[key]["q01"]), 8)
                self.assertEqual(len(stats[key]["q99"]), 8)
            dataset = LeRobotDataset(repo_id="local/synthetic_fr3", root=output, video_backend="pyav")
            self.assertEqual(len(dataset), fixture.N)
            for index in (0, fixture.N - 1):
                sample = dataset[index]
                for key in ("observation.images.camera_1", "observation.images.camera_2"):
                    self.assertEqual(tuple(sample[key].shape), (3, 480, 640))
                    self.assertTrue(fixture.np.isfinite(fixture.np.asarray(sample[key])).all())
            preflight = subprocess.run([
                sys.executable, str(ROOT / "check_env.py"), "--dataset", str(output), "--data-only",
            ], env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
            checked = json.loads(preflight.stdout)
            self.assertEqual(checked["status"], "passed")
            self.assertEqual(checked["dataset"]["frames"], fixture.N)
            self.assertTrue(checked["dataset"]["decoded_samples"][-1]["end_padding_verified"])


if __name__ == "__main__":
    unittest.main()
