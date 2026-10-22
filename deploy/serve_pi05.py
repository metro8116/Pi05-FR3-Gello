#!/usr/bin/env python3
"""AutoDL entry point. Bind loopback and expose through a local SSH tunnel."""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import time
import uuid
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get('PI05_CHECKPOINT'),
                        help="Local official checkpoint/pretrained_model directory; or PI05_CHECKPOINT")
    parser.add_argument("--train-project", default=str(Path(__file__).resolve().parents[1]),
                        help="Repository root containing the shared fr3_pi05 runtime (no dataset required)")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "::1"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--tokenizer", help="Optional existing local directory/cached ID; otherwise use saved processor")
    parser.add_argument("--task", default="Put the avocado in the basket.",
                        choices=("Put the avocado in the basket.",))
    parser.add_argument("--warmup", type=int, default=1, help="Synthetic forward passes before accepting clients")
    parser.add_argument("--client-timeout", type=float, default=30.0)
    parser.add_argument("--check-only", action="store_true", help="Check local assets/environment without allocating PI05")
    parser.add_argument("--mock", action="store_true", help="Synthetic communication test; real robot arming is prohibited")
    return parser


def make_handler(service, client_timeout=30.0):
    from protocol import PROTOCOL_VERSION, ProtocolError, recv_packet, send_packet
    from server_runtime import ServerBusyError

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(client_timeout)
            last_request_id = -1
            session_id = None
            while True:
                packet = None
                try:
                    packet = recv_packet(self.request)
                    if packet is None:
                        return
                    if packet.get("version") != PROTOCOL_VERSION:
                        raise ProtocolError("Unsupported protocol version")
                    if packet.get("type") == "hello":
                        response = service.metadata()
                    elif packet.get("type") == "ping":
                        response = {"version": PROTOCOL_VERSION, "type": "pong", "server_monotonic": time.monotonic()}
                    elif packet.get("type") == "infer":
                        request_id = packet.get("request_id")
                        # Request IDs are monotonically increasing per connection;
                        # duplicate network requests never replay an old action block.
                        if type(request_id) is not int or request_id <= last_request_id:
                            raise ProtocolError("request_id must increase on this connection")
                        if session_id is not None and packet.get("session_id") != session_id:
                            raise ProtocolError("session_id cannot change within a connection")
                        response = service.infer(packet)
                        session_id, last_request_id = packet["session_id"], request_id
                    else:
                        raise ProtocolError("Unknown request type")
                    send_packet(self.request, response)
                except (EOFError, ConnectionError, socket.timeout, OSError):
                    return
                except Exception as exc:
                    # Structured errors carry sequence context but never include
                    # input images, tensors, credentials, or arbitrary tracebacks.
                    response = {"version": PROTOCOL_VERSION, "type": "error",
                                "code": "busy" if isinstance(exc, ServerBusyError) else "invalid_or_failed_request",
                                "message": str(exc)[:1000]}
                    if isinstance(packet, dict):
                        for key in ("session_id", "request_id", "capture_seq"):
                            value = packet.get(key)
                            if isinstance(value, (str, int)):
                                response[key] = value
                    try:
                        send_packet(self.request, response)
                    except (ConnectionError, OSError):
                        return
                    # A malformed framing error may leave unread payload bytes.
                    # Close that connection rather than trying to resynchronize.
                    if packet is None:
                        return

    return Handler


class ThreadedInferenceServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 4


def warmup(service, passes: int):
    if passes == 0:
        return
    import numpy as np
    from protocol import CAMERA_KEYS, PROTOCOL_VERSION, encode_image

    image = encode_image(np.zeros((480, 640, 3), dtype=np.uint8))
    for index in range(passes):
        request = {"version": PROTOCOL_VERSION, "type": "infer", "session_id": str(uuid.uuid4()),
                   "request_id": index, "capture_seq": index, "capture_monotonic": time.monotonic(),
                   "observation": {"state": [0., -0.7, 0., -2., 0., 1.3, 0., 0.04],
                                   "images": {key: image for key in CAMERA_KEYS}, "task": service.task}}
        response = service.infer(request)
        print(f"Warmup {index + 1}/{passes}: {response['inference_s']:.3f}s (synthetic inputs; no robot)", flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535 or not 0 <= args.warmup <= 20 or args.client_timeout <= 0:
        raise SystemExit("Invalid port, warmup or client timeout")
    if not args.mock and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --mock is explicitly selected")
    from server_runtime import InferenceService, MockPolicyRuntime, load_official_runtime

    try:
        runtime = MockPolicyRuntime() if args.mock else load_official_runtime(
            args.checkpoint, args.train_project, device=args.device, dtype=args.dtype,
            tokenizer=args.tokenizer, check_only=args.check_only)
        if args.check_only:
            print(json.dumps(runtime if isinstance(runtime, dict) else InferenceService(runtime, args.task).metadata(),
                             indent=2, ensure_ascii=False))
            return 0
        service = InferenceService(runtime, args.task)
        warmup(service, args.warmup)
        if args.host == "::1":
            ThreadedInferenceServer.address_family = socket.AF_INET6
        with ThreadedInferenceServer((args.host, args.port), make_handler(service, args.client_timeout)) as server:
            print(json.dumps(service.metadata(), indent=2, ensure_ascii=False), flush=True)
            print(f"PI05 ready at {args.host}:{args.port}; use SSH forwarding. Ctrl+C stops the service.", flush=True)
            try:
                server.serve_forever(poll_interval=0.25)
            except KeyboardInterrupt:
                print("Stopping PI05 inference server.", flush=True)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
