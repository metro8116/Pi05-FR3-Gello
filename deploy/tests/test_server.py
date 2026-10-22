"""CPU/offline server tests. Never connect to AutoDL, ROS or a robot."""
from __future__ import annotations

import copy
import importlib.util
import json
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import CAMERA_KEYS, TASK, ProtocolError, encode_image, recv_packet, send_packet, validate_metadata
from serve_pi05 import ThreadedInferenceServer, build_parser, make_handler, warmup
from server_runtime import (InferenceService, MockPolicyRuntime, OfficialPolicyRuntime,
                            ServerBusyError, inspect_checkpoint)


def request_packet():
    image = encode_image(np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8))
    return {"version": 1, "type": "infer", "session_id": str(uuid.uuid4()), "request_id": 1,
            "capture_seq": 5, "capture_monotonic": 1234.5,
            "observation": {"state": [0., -0.7, 0., -2., 0., 1.3, 0., 0.04],
                            "images": {key: image for key in CAMERA_KEYS}, "task": TASK}}


def write_tiny_safetensors(path):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0))


def checkpoint_fixture(directory):
    directory.mkdir(parents=True)
    config = {"type": "pi05", "chunk_size": 50, "n_obs_steps": 1, "use_relative_actions": False,
              "input_features": {"observation.state": {"shape": [8]},
                                 **{key: {"shape": [3, 480, 640]} for key in CAMERA_KEYS}},
              "output_features": {"action": {"shape": [8]}},
              "normalization_mapping": {"STATE": "QUANTILES", "ACTION": "QUANTILES"}}
    (directory / "config.json").write_text(json.dumps(config))
    for filename in ("model.safetensors", "normalizer.safetensors", "unnormalizer.safetensors"):
        write_tiny_safetensors(directory / filename)
    pre = {"steps": [{"registry_name": "normalizer_processor", "config": {},
                      "state_file": "normalizer.safetensors"},
                     {"registry_name": "tokenizer_processor", "config": {"tokenizer_name": "cached/tokenizer"}},
                     {"registry_name": "relative_actions_processor", "config": {"enabled": False}}]}
    post = {"steps": [{"registry_name": "unnormalizer_processor", "config": {},
                       "state_file": "unnormalizer.safetensors"}]}
    (directory / "policy_preprocessor.json").write_text(json.dumps(pre))
    (directory / "policy_postprocessor.json").write_text(json.dumps(post))
    return directory


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = checkpoint_fixture(self.root / "checkpoints/002000/pretrained_model")

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoint_assets_preflight_has_no_model_allocation(self):
        report = inspect_checkpoint(self.path.parent)
        self.assertEqual(report["checkpoint"], str(self.path))
        self.assertEqual(report["tensor_count"], 1)
        self.assertEqual(report["tokenizer"], "cached/tokenizer")

    def test_pins_last_symlink_and_identity_changes_per_checkpoint(self):
        alias = self.path.parent.parent / "last"
        alias.symlink_to(self.path.parent)
        report = inspect_checkpoint(alias)
        self.assertEqual(report["checkpoint"], str(self.path))
        next_path = checkpoint_fixture(self.path.parent.parent / "004000/pretrained_model")
        alias.unlink()
        alias.symlink_to(next_path.parent)
        self.assertEqual(report["checkpoint"], str(self.path))
        self.assertNotEqual(inspect_checkpoint(alias)["model_id"], report["model_id"])

    def test_saved_stats_missing_is_fatal_not_recomputed(self):
        (self.path / "normalizer.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            inspect_checkpoint(self.path)

    def test_wrong_state_action_or_relative_contract_is_rejected(self):
        config_path = self.path / "config.json"
        original = json.loads(config_path.read_text())
        for field, value in (("use_relative_actions", True), ("chunk_size", 20)):
            config = {**original, field: value}
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                inspect_checkpoint(self.path)
        config = copy.deepcopy(original)
        config["output_features"]["action"]["shape"] = [32]
        config_path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            inspect_checkpoint(self.path)

    def test_processor_path_escape_and_relative_actions_rejected(self):
        file = self.path / "policy_preprocessor.json"
        original = json.loads(file.read_text())
        original["steps"][-1]["config"]["enabled"] = True
        file.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "relative"):
            inspect_checkpoint(self.path)
        original["steps"][-1]["config"]["enabled"] = False
        original["steps"][0]["state_file"] = "../outside.safetensors"
        file.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "inside"):
            inspect_checkpoint(self.path)

    def test_truncated_model_fails_before_loading(self):
        file = self.path / "model.safetensors"
        file.write_bytes(file.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "offsets"):
            inspect_checkpoint(self.path)


class InferenceTests(unittest.TestCase):
    def test_mock_identification_and_echo(self):
        service = InferenceService(MockPolicyRuntime())
        self.assertTrue(validate_metadata(service.metadata())["mock"])
        request = request_packet()
        response = service.infer(request)
        self.assertEqual(len(response["actions"]), 50)
        self.assertEqual(response["actions"][0], request["observation"]["state"])
        for key in ("session_id", "request_id", "capture_seq", "capture_monotonic"):
            self.assertEqual(response[key], request[key])

    def test_busy_never_queues_old_observation(self):
        service = InferenceService(MockPolicyRuntime())
        with service.model_lock:
            with self.assertRaises(ServerBusyError):
                service.infer(request_packet())
        self.assertEqual(service.infer(request_packet())["type"], "actions")

    def test_malformed_images_and_nonfinite_state_rejected(self):
        service = InferenceService(MockPolicyRuntime())
        request = request_packet()
        request["observation"]["images"][CAMERA_KEYS[0]]["data"] = "YWJj"
        with self.assertRaises(ProtocolError):
            service.infer(request)
        request = request_packet()
        request["observation"]["state"][0] = float("nan")
        with self.assertRaises(ProtocolError):
            service.infer(request)

    def test_inference_error_releases_model_lock(self):
        runtime = MockPolicyRuntime()
        service = InferenceService(runtime)
        with patch.object(runtime, "predict", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                service.infer(request_packet())
        self.assertFalse(service.model_lock.locked())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not available")
    def test_tiny_policy_official_call_order_rgb_and_raw_action_units(self):
        import torch

        calls = []

        def preprocess(batch):
            calls.append("preprocessor")
            self.assertEqual(tuple(batch["observation.state"].shape), (8,))
            self.assertEqual(batch["observation.state"][-1].item(), np.float32(0.04))
            image = batch[CAMERA_KEYS[0]]
            self.assertEqual(tuple(image.shape), (3, 480, 640))
            self.assertGreater(image[0].mean().item(), 0.98)
            self.assertLess(image[2].mean().item(), 0.02)
            self.assertEqual(batch["task"], TASK)
            return {"already_preprocessed": True}

        class TinyPolicy:
            def predict_action_chunk(self, batch):
                calls.append("policy.predict_action_chunk")
                assert batch == {"already_preprocessed": True}
                return torch.zeros(1, 50, 8)

        def postprocess(actions):
            calls.append("postprocessor")
            result = actions + 0.3
            result[..., 7] = 0.025
            return result

        runtime = OfficialPolicyRuntime(TinyPolicy(), preprocess, postprocess,
                                        checkpoint="local", model_id="tiny", device="cpu", dtype="float32")
        result = InferenceService(runtime).infer(request_packet())
        self.assertEqual(calls, ["preprocessor", "policy.predict_action_chunk", "postprocessor"])
        self.assertAlmostEqual(result["actions"][0][7], 0.025)
        self.assertAlmostEqual(result["actions"][0][0], 0.3)

    def test_warmup_is_synthetic_and_no_hardware(self):
        service = InferenceService(MockPolicyRuntime())
        warmup(service, 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not available")
    def test_cuda_bf16_selects_explicit_autocast_without_starting_gpu(self):
        import torch

        class TinyPolicy:
            def predict_action_chunk(self, batch):
                return torch.zeros(1, 50, 8)

        runtime = OfficialPolicyRuntime(TinyPolicy(), lambda value: value, lambda value: value,
                                        checkpoint="tiny", model_id="tiny", device="cuda", dtype="bfloat16")
        with patch.object(torch, "autocast") as autocast:
            InferenceService(runtime).infer(request_packet())
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)


class TCPServerTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadedInferenceServer(("127.0.0.1", 0), make_handler(InferenceService(MockPolicyRuntime()), 2))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.client = socket.create_connection(self.server.server_address, timeout=3)

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def test_hello_ping_infer_and_repeated_request_rejected(self):
        send_packet(self.client, {"version": 1, "type": "hello"})
        self.assertTrue(validate_metadata(recv_packet(self.client))["mock"])
        send_packet(self.client, {"version": 1, "type": "ping"})
        self.assertEqual(recv_packet(self.client)["type"], "pong")
        request = request_packet()
        send_packet(self.client, request)
        self.assertEqual(recv_packet(self.client)["type"], "actions")
        send_packet(self.client, request)
        self.assertEqual(recv_packet(self.client)["type"], "error")

    def test_huge_packet_gets_error_and_connection_closes(self):
        self.client.sendall(struct.pack("!I", 9 * 1024 * 1024))
        self.assertEqual(recv_packet(self.client)["type"], "error")
        with self.assertRaises(EOFError):
            recv_packet(self.client)

    def test_malformed_image_returns_error_then_next_request_works(self):
        request = request_packet()
        request["observation"]["images"][CAMERA_KEYS[0]]["data"] = "YmFk"
        send_packet(self.client, request)
        self.assertEqual(recv_packet(self.client)["type"], "error")
        request = request_packet()
        send_packet(self.client, request)
        self.assertEqual(recv_packet(self.client)["type"], "actions")

    def test_help_imports_without_lerobot_model(self):
        args = build_parser().parse_args(["--mock", "--warmup", "0"])
        self.assertTrue(args.mock)
        self.assertEqual(args.host, "127.0.0.1")


class OfficialProcessorIntegrationTests(unittest.TestCase):
    """Exercise installed official CPU processors with a local tiny tokenizer.

    The local data environment is legacy 0.4.4, so this verifies the saved
    processor tensor contract, not a complete AutoDL 0.6.1 PI05 GPU forward.
    """

    def test_saved_official_processors_roundtrip_without_dataset_or_large_model(self):
        try:
            import torch
            from tokenizers import Tokenizer
            from tokenizers.models import WordLevel
            from transformers import PreTrainedTokenizerFast
            from lerobot.configs.types import NormalizationMode
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            from lerobot.processor import PolicyProcessorPipeline
            from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
                                                     policy_action_to_transition, transition_to_policy_action)
        except ImportError as exc:
            self.skipTest(str(exc))
        train_src = Path(__file__).resolve().parents[2] / "src"
        sys.path.insert(0, str(train_src))
        from fr3_pi05 import runtime as training_runtime

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tokenizer"
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
                pad_token="[PAD]", unk_token="[UNK]")
            tokenizer.save_pretrained(source)
            cfg = PI05Config(device="cpu", dtype="float32")
            cfg.input_features, cfg.output_features = training_runtime._fr3_features()
            cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY,
                                         "STATE": NormalizationMode.QUANTILES,
                                         "ACTION": NormalizationMode.QUANTILES}
            stats = {key: {"q01": torch.tensor([-3.] * 7 + [0.]),
                           "q99": torch.tensor([3.] * 7 + [.08])}
                     for key in ("observation.state", "action")}
            pre, post = training_runtime.make_processors(cfg, stats, source, "cpu")
            pre.save_pretrained(root, config_filename="policy_preprocessor.json")
            post.save_pretrained(root, config_filename="policy_postprocessor.json")
            loaded_pre = PolicyProcessorPipeline.from_pretrained(
                root, config_filename="policy_preprocessor.json", local_files_only=True,
                to_transition=batch_to_transition, to_output=transition_to_batch)
            loaded_post = PolicyProcessorPipeline.from_pretrained(
                root, config_filename="policy_postprocessor.json", local_files_only=True,
                to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

            class TinyPolicy:
                def predict_action_chunk(inner_self, batch):
                    self.assertEqual(tuple(batch["observation.state"].shape), (1, 8))
                    self.assertEqual(tuple(batch[CAMERA_KEYS[0]].shape), (1, 3, 480, 640))
                    self.assertEqual(tuple(batch["observation.language.tokens"].shape), (1, 200))
                    return torch.full((1, 50, 8), .5)

            runtime = OfficialPolicyRuntime(TinyPolicy(), loaded_pre, loaded_post,
                                            checkpoint="tiny-cpu", model_id="tiny", device="cpu", dtype="float32")
            response = InferenceService(runtime).infer(request_packet())
            self.assertAlmostEqual(response["actions"][0][0], 1.5, places=5)
            self.assertAlmostEqual(response["actions"][0][7], 0.06, places=5)


if __name__ == "__main__":
    unittest.main()
