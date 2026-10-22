"""Repository portability and shared implementation checks, without hardware."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = DEPLOY_ROOT.parent
sys.path.insert(0, str(DEPLOY_ROOT))

import checkpoint_loading
import config
import control_ownership
import local_session
import prepare_local
from serve_pi05 import build_parser
from common import fr3_control_ownership
from fr3_pi05 import runtime as training_runtime


class RepositoryLayoutTests(unittest.TestCase):
    def test_workspace_container_and_config_defaults_belong_to_this_repository(self):
        self.assertEqual(config.ROOT, DEPLOY_ROOT)
        self.assertEqual(config.REPO_ROOT, REPOSITORY)
        self.assertEqual(config.DEFAULT_CONFIG, DEPLOY_ROOT / 'configs/deploy.yaml')
        self.assertEqual(prepare_local.HOST_WS, REPOSITORY / 'ros2_ws')
        self.assertEqual(prepare_local.GRIPPER_SOURCE, REPOSITORY / 'ros2_ws/src/pi05_gripper_driver')
        self.assertTrue((prepare_local.GRIPPER_SOURCE / 'LICENSE').is_file())
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(prepare_local.container_name(), 'pi05-fr3-gello-ros2-1')

    def test_control_lease_and_native_loader_are_the_shared_implementations(self):
        self.assertIs(control_ownership, fr3_control_ownership)
        self.assertIs(control_ownership.ControlLease, fr3_control_ownership.ControlLease)
        self.assertIs(checkpoint_loading.load_native_weights_strict,
                      training_runtime.load_native_weights_strict)

    def test_server_defaults_to_repository_training_code_and_checkpoint_environment(self):
        with patch.dict(os.environ, {'PI05_CHECKPOINT': 'checkpoints/example/pretrained_model'}):
            args = build_parser().parse_args([])
        self.assertEqual(Path(args.train_project), REPOSITORY)
        self.assertEqual(args.checkpoint, 'checkpoints/example/pretrained_model')

    def test_bundled_bridge_help_needs_no_host_repo_pythonpath(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'deploy'
            source.mkdir()
            for path in DEPLOY_ROOT.glob('*.py'):
                shutil.copy2(path, source / path.name)
            for relative in ('ros2/pi05_fr3.launch.py', 'configs/controllers_pi05.yaml'):
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(DEPLOY_ROOT / relative, target)
            bundle = root / 'bundle'
            bundle.mkdir()
            remote = '/tmp/pi05-deploy-portable-test'

            def fake_docker(argv, **kwargs):
                self.assertEqual(argv[0], 'mock-docker')
                if argv[1] == 'exec':
                    self.assertEqual(argv[2:4], ['mock-container', 'mkdir'])
                else:
                    self.assertEqual(argv[1], 'cp')
                    target = argv[3].removeprefix('mock-container:' + remote + '/')
                    self.assertNotEqual(target, argv[3])
                    output = bundle / target
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(argv[2], output)
                return subprocess.CompletedProcess(argv, 0)

            with patch.object(local_session, 'ROOT', source), \
                    patch.object(local_session, 'docker_base', return_value=['mock-docker']), \
                    patch.object(local_session, 'container_name', return_value='mock-container'), \
                    patch.object(local_session.subprocess, 'run', side_effect=fake_docker):
                self.assertEqual(local_session.bundle_for_container(config.load_config(), 'portable-test'), remote)
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            completed = subprocess.run([sys.executable, '-B', str(bundle / 'fr3_policy_bridge.py'), '--help'],
                                       cwd=root, env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('--config', completed.stdout)


if __name__ == '__main__':
    unittest.main()
