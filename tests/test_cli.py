"""Native trainer arguments and checkpoint compatibility, without importing the model."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fr3_pi05.cli import DEFAULTS, main, native_arguments, parser, resolve_config


def make_native_checkpoint(root):
    root = Path(root)
    policy = {"type": "pi05", "input_features": {
        "observation.state": {"type": "STATE", "shape": [8]},
        "observation.images.camera_1": {"type": "VISUAL", "shape": [3,480,640]},
        "observation.images.camera_2": {"type": "VISUAL", "shape": [3,480,640]}},
        "output_features": {"action": {"type": "ACTION", "shape": [8]}},
        "use_relative_actions": False, "dtype": "bfloat16", "device": "cuda",
        "gradient_checkpointing": False, "freeze_vision_encoder": True, "train_expert_only": False}
    saved = {**DEFAULTS, "policy": policy,
             "dataset": {"root": "/old/data", "repo_id": "local/fr3_avocado_pi05", "eval_split": 0.0, "episodes": None},
             "output_dir": "/old/output", "wandb": {"enable": True, "project": "fr3-avocado-pi05"}}
    pre = root / "pretrained_model"
    pre.mkdir(parents=True)
    (pre / "train_config.json").write_text(json.dumps(saved))
    (pre / "config.json").write_text(json.dumps(policy))
    (pre / "model.safetensors").touch()
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (pre / name).write_text('{"steps": []}')
    state = root / "training_state"
    state.mkdir()
    for name in ("optimizer_param_groups.json", "optimizer_state.safetensors", "rng_state.safetensors", "scheduler_state.json", "training_step.json"):
        (state / name).write_text("{}")
    return saved


class CLITests(unittest.TestCase):
    def test_experiment_defaults_and_native_flags(self):
        config = resolve_config(parser().parse_args(["--base-model", "/tmp/pi05_base"]))
        for key, value in DEFAULTS.items():
            self.assertEqual(config[key], value, key)
        flags = native_arguments(config, "/tmp/prepared")
        required = {"--policy.path=/tmp/prepared", "--steps=20000", "--batch_size=16", "--num_workers=8",
                    "--save_freq=2000", "--log_freq=10", "--policy.gradient_checkpointing=false",
                    "--policy.freeze_vision_encoder=true", "--policy.train_expert_only=false",
                    "--dataset.eval_split=0.0", "--eval_steps=0", "--env_eval_freq=0", "--dataset.episodes=null",
                    "--policy.use_relative_actions=false", "--policy.optimizer_lr=2.5e-05",
                    "--policy.scheduler_warmup_steps=1000", "--policy.scheduler_decay_steps=30000",
                    "--wandb.disable_artifact=true", "--policy.push_to_hub=false"}
        self.assertTrue(required.issubset(flags), required-set(flags))
        self.assertFalse(config["validation"])
        self.assertNotIn("--resume=true", flags)
        self.assertFalse(any("gradient_accumulation" in f for f in flags))

    def test_old_project_parameter_aliases(self):
        args = parser().parse_args(["--policy-path", "lerobot/pi05_base", "--data-root", "./pick_place_data", "--offline",
                                   "--gradient-checkpointing", "false", "--batch-size", "16", "--num-workers", "8",
                                   "--steps", "20000", "--save-freq", "2000", "--log-freq", "10", "--wandb", "true", "--out", "/tmp/run"])
        config = resolve_config(args)
        self.assertEqual(config["base_model"], "lerobot/pi05_base")
        self.assertEqual(config["output_dir"], "/tmp/run")
        self.assertTrue(config["wandb"])
        self.assertEqual(Path(config["dataset"]).name, "pick_place_data")

    def test_dry_run_does_not_import_torch_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            code = "import runpy, sys; sys.argv=" + repr([str(ROOT / "train.py"), "--dry-run", "--out", str(output)]) + ";\ntry: runpy.run_path(sys.argv[0], run_name='__main__')\nexcept SystemExit as e:\n assert e.code==0\nassert 'torch' not in sys.modules\nassert 'transformers' not in sys.modules\nassert 'fr3_pi05.native' not in sys.modules\n"
            run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('"trainer": "lerobot.scripts.lerobot_train.train"', run.stdout)
            self.assertFalse(output.exists())

    def test_resume_uses_native_config_never_policy_path_and_preserves_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoints/000001"
            make_native_checkpoint(checkpoint)
            for suffix in ([], ["--resume-from", str(checkpoint / "pretrained_model")]):
                argv = ["--resume", *suffix] if suffix else ["--resume", str(checkpoint)]
                config = resolve_config(parser().parse_args([*argv, "--out", "/tmp/resume", "--data-root", "/relocated/data"]))
                flags = native_arguments(config)
                self.assertIn("--resume=true", flags)
                self.assertIn(f"--config_path={checkpoint}/pretrained_model/train_config.json", flags)
                self.assertIn("--output_dir=/tmp/resume", flags)
                self.assertIn("--dataset.root=/relocated/data", flags)
                self.assertTrue(config["wandb"])
                self.assertFalse(any(f.startswith("--policy.path=") for f in flags))
                self.assertFalse(any(f.startswith("--steps=") for f in flags))
            with self.assertRaisesRegex(ValueError, "preserves steps"):
                resolve_config(parser().parse_args(["--resume", str(checkpoint), "--steps", "22000"]))
            with self.assertRaisesRegex(ValueError, "omit --base-model"):
                resolve_config(parser().parse_args(["--resume", str(checkpoint), "--policy-path", "lerobot/pi05_base"]))

    def test_incomplete_or_custom_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "COMPLETE.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "old custom-loop"):
                resolve_config(parser().parse_args(["--resume", tmp]))
        with tempfile.TemporaryDirectory() as tmp:
            make_native_checkpoint(tmp)
            Path(tmp, "training_state/rng_state.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "rng_state"):
                resolve_config(parser().parse_args(["--resume", tmp]))

    def test_simulation_or_relative_checkpoint_is_rejected(self):
        for change in ("dim", "relative", "split", "processor"):
            with tempfile.TemporaryDirectory() as tmp:
                saved = make_native_checkpoint(tmp)
                if change == "dim":
                    saved["policy"]["input_features"]["observation.state"]["shape"] = [7]
                elif change == "relative":
                    saved["policy"]["use_relative_actions"] = True
                elif change == "split":
                    saved["dataset"]["eval_split"] = 0.1
                else:
                    Path(tmp, "pretrained_model/policy_preprocessor.json").write_text('{"steps":[{"state_file":"missing.safetensors"}]}')
                Path(tmp, "pretrained_model/train_config.json").write_text(json.dumps(saved))
                with self.assertRaises(ValueError):
                    resolve_config(parser().parse_args(["--resume", tmp]))

    def test_invalid_hyperparameters(self):
        for extra in (["--steps", "0"], ["--batch-size", "0"], ["--lr", "nan"], ["--weight-decay", "inf"],
                      ["--num-workers", "-1"], ["--n-action-steps", "51"], ["--gradient-accumulation-steps", "2"]):
            with self.assertRaises(ValueError, msg=extra):
                resolve_config(parser().parse_args(extra))

    def test_entry_dispatches_to_native_launch(self):
        with patch("fr3_pi05.native.launch") as launch, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--steps", "2"]), 0)
        self.assertEqual(launch.call_args.args[0]["steps"], 2)


if __name__ == "__main__":
    unittest.main()
