"""Exercise entry points in a relocated checkout without importing hardware code."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def make_checkout(self, directory):
        repo = Path(directory) / "relocated repository with spaces"
        (repo / "configs").mkdir(parents=True)
        (repo / "scripts").mkdir()
        for relative in ("run.sh", "scripts/setup_env.sh", "scripts/doctor.py"):
            shutil.copy2(ROOT / relative, repo / relative)
        spy = repo / "spy python"
        spy.write_text("#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps({'args':sys.argv[1:],'root':os.environ['PI05_REPO_ROOT'],'path':os.environ['PYTHONPATH']}))\n")
        spy.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PI05_")}
        for profile in ("ROBOT", "CONVERTER", "ML"):
            env[f"PI05_{profile}_PYTHON"] = str(spy)
        return repo, env

    def test_relocated_paths_preserve_arguments_and_select_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.make_checkout(directory)
            for command, expected in (("validate", "tools/convert_to_lerobot_v30.py"),
                                      ("train", "train.py"), ("resume", "train.py"),
                                      ("serve", "deploy/serve_pi05.py"),
                                      ("prepare", "deploy/local_session.py"),
                                      ("deploy", "deploy/local_session.py"),
                                      ("stop-deploy", "deploy/local_session.py")):
                result = subprocess.run(["bash", str(repo / "run.sh"), command, "--example", "a path with spaces"],
                                        cwd=directory, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(result.stdout)
                self.assertEqual(record["args"][0], str(repo / expected))
                self.assertEqual(record["args"][-2:], ["--example", "a path with spaces"])
                self.assertEqual(record["root"], str(repo))
                self.assertIn(str(repo / "src"), record["path"].split(":"))
                for name, flag in (("validate", "--dry-run"), ("resume", "--resume"),
                                   ("prepare", "--prepare"), ("stop-deploy", "--stop")):
                    if command == name:
                        self.assertIn(flag, record["args"])

    def test_machine_file_is_used_without_rewriting_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.make_checkout(directory)
            spy = repo / "spy python"
            config = repo / "configs/autodl.env"
            config.write_text('export PI05_ML_PYTHON="$PI05_REPO_ROOT/spy python"\n')
            env["PI05_ML_PYTHON"] = "/missing/environment"
            before = {p.name: p.read_bytes() for p in (spy, config)}
            result = subprocess.run(["bash", str(repo / "run.sh"), "train", "--dry-run"], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["args"][-1], "--dry-run")
            self.assertEqual(before, {p.name: p.read_bytes() for p in (spy, config)})
            self.assertFalse((repo / ".venv-ml").exists())

    def test_setup_refuses_existing_environment_before_invoking_uv(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.make_checkout(directory)
            existing = repo / ".venv-ml"
            existing.mkdir()
            sentinel = existing / "do-not-change"
            sentinel.write_text("existing state")
            bin_dir = Path(directory) / "fake-bin"
            bin_dir.mkdir()
            uv = bin_dir / "uv"
            uv.write_text("#!/bin/sh\necho 'uv must never be called' >&2\nexit 99\n")
            uv.chmod(0o755)
            env["PATH"] = str(bin_dir) + ":" + os.environ["PATH"]
            result = subprocess.run(["bash", str(repo / "run.sh"), "setup", "ml"], env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Refusing to modify existing environment", result.stderr)
            self.assertNotIn("uv must never", result.stderr)
            self.assertEqual(sentinel.read_text(), "existing state")

    def test_doctor_never_requires_robot_or_ml_imports(self):
        result = subprocess.run(["bash", str(ROOT / "run.sh"), "doctor"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["hardware_started"])

    def test_resume_only_installs_into_the_selected_repository_venv(self):
        version = sys.version_info[:2]
        if version not in ((3, 10), (3, 12)):
            self.skipTest("Setup supports Python 3.10 robot or Python 3.12 converter/ML")
        profile = "robot" if version == (3, 10) else "converter"
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.make_checkout(directory)
            target = repo / (".venv-" + profile)
            venv.EnvBuilder(with_pip=False).create(target)
            sentinel = target / "preserve-this-file"
            sentinel.write_text("unchanged")
            bin_dir = Path(directory) / "fake-bin"
            bin_dir.mkdir()
            calls = Path(directory) / "uv-calls.jsonl"
            uv = bin_dir / "uv"
            uv.write_text("#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ['TEST_UV_CALLS'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
            uv.chmod(0o755)
            env.update(PATH=str(bin_dir) + ":" + os.environ["PATH"], TEST_UV_CALLS=str(calls))
            result = subprocess.run(["bash", str(repo / "run.sh"), "setup", profile, "--resume"],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(commands), 3)
            for command in commands[:2]:
                self.assertEqual(command[:4], ["pip", "install", "--python", str(target / "bin/python")])
            self.assertEqual(commands[-1], ["pip", "check", "--python", str(target / "bin/python")])
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_resume_rejects_foreign_environment_symlinks_before_install(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, env = self.make_checkout(directory)
            foreign = Path(directory) / "existing-training-environment"
            foreign.mkdir()
            (foreign / "keep").write_text("do not touch")
            (repo / ".venv-ml").symlink_to(foreign, target_is_directory=True)
            # The stub only allows the tool-presence check; installation would
            # create the marker, which must remain absent.
            bin_dir = Path(directory) / "fake-bin"
            bin_dir.mkdir()
            marker = Path(directory) / "uv-was-called"
            uv = bin_dir / "uv"
            uv.write_text('#!/bin/sh\ntouch "$TEST_UV_MARKER"\nexit 99\n')
            uv.chmod(0o755)
            env.update(PATH=str(bin_dir) + ":" + os.environ["PATH"], TEST_UV_MARKER=str(marker))
            result = subprocess.run(["bash", str(repo / "run.sh"), "setup", "ml", "--resume"],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("symlink", result.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual((foreign / "keep").read_text(), "do not touch")


if __name__ == "__main__":
    unittest.main()
