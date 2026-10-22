"""Doctor behavior and tiny safetensors-header checks, without a large model."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fr3_pi05 import doctor


class DoctorTests(unittest.TestCase):
    def test_parser_reads_existing_asset_environment(self):
        with patch.dict(os.environ, {"PI05_BASE_MODEL": "/old/base", "PI05_TOKENIZER": "/old/tok"}):
            args = doctor.parser().parse_args(["--data-only"])
        self.assertEqual(args.base_model, "/old/base")
        self.assertEqual(args.tokenizer, "/old/tok")
        self.assertTrue(args.data_only)

    def test_cli_reports_check_failures_without_training(self):
        with patch.object(doctor, "run_checks", side_effect=ValueError("incomplete input")):
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                result = doctor.main(["--data-only"])
        self.assertEqual(result, 1)
        self.assertIn("ERROR: incomplete input", output.getvalue())

    def test_cli_outputs_valid_json(self):
        with patch.object(doctor, "run_checks", return_value={"validation": False, "frames": 12495}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = doctor.main(["--data-only"])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), {"validation": False, "frames": 12495})

    def test_weight_header_does_not_construct_policy(self):
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "config.json").write_text('{"type":"pi05"}')
            save_file({"model.layer.weight": torch.zeros(2, 3)}, str(directory / "model.safetensors"))
            with patch("fr3_pi05.runtime.load_weights_strict", side_effect=AssertionError("must not allocate")):
                report = doctor.inspect_weight_header(directory)
        self.assertEqual(report["tensor_count"], 1)
        self.assertEqual(report["total_tensor_elements"], 6)
        self.assertIn("not performed", report["full_strict_weight_load"])

    def test_data_only_does_not_import_or_call_runtime(self):
        contract = {
            "num_frames": 12495, "num_episodes": 50, "fps": 15,
            "task": "Put the avocado in the basket.", "ignored_split_sidecar": True,
        }
        stats = {key: {"count": [12495], **{name: [0.0]*8 for name in ("mean", "std", "min", "max", "q01", "q99")}} for key in ("observation.state", "action")}
        args = doctor.parser().parse_args(["--data-only"])
        unusable_runtime = types.ModuleType("fr3_pi05.runtime")
        with patch.dict(sys.modules, {"fr3_pi05.runtime": unusable_runtime}), \
                patch("fr3_pi05.data.inspect_contract", return_value=contract), \
                patch("fr3_pi05.data.make_dataset", return_value=object()), \
                patch("fr3_pi05.data.compute_training_stats", return_value=stats), \
                patch("fr3_pi05.native.read_task_stats", return_value=stats), \
                patch.object(doctor, "_check_decoded_samples", return_value=[]):
            summary = doctor.run_checks(args)
        self.assertEqual(summary["runtime"], "skipped (--data-only)")
        self.assertEqual(summary["dataset"]["episodes"], 50)
        self.assertFalse(summary["dataset"]["validation"])
        self.assertFalse(summary["training_defaults"]["gradient_checkpointing"])


if __name__ == "__main__":
    unittest.main()
