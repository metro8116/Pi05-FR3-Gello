"""Bounded, versioned JSON packets over local TCP / an SSH tunnel; never pickle."""
from __future__ import annotations

import base64
import binascii
import io
import json
import math
import socket
import struct
import uuid

VERSION = PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 8 * 1024 * 1024
MAX_JPEG_BYTES = 2 * 1024 * 1024
CAMERA_KEYS = ('observation.images.camera_1', 'observation.images.camera_2')
TASK = 'Put the avocado in the basket.'
ACTION_SEMANTICS = 'absolute_joint_positions_rad_and_total_gripper_width_m'


class ProtocolError(ValueError):
    pass


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f'{name} must be finite numeric data')
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ProtocolError(f'{name} must be a nonnegative integer')
    return value


def _vector(values, length, name):
    if not isinstance(values, list) or len(values) != length:
        raise ProtocolError(f'{name} must be a {length}-element list')
    for value in values:
        _number(value, name)
    return values


def _session(value):
    if not isinstance(value, str) or len(value) > 36:
        raise ProtocolError('session_id must be a UUID string')
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError('session_id must be a UUID string') from exc


def _json_constant(value):
    raise ProtocolError(f'Non-finite JSON constant: {value}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f'Duplicate JSON field: {key}')
        result[key] = value
    return result


def _read_exact(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        block = sock.recv(count - len(chunks))
        if not block:
            if not chunks:
                raise EOFError('Connection closed')
            raise ProtocolError('Truncated packet')
        chunks.extend(block)
    return bytes(chunks)


def send_packet(sock, packet):
    if not isinstance(packet, dict):
        raise ProtocolError('Packet must be an object')
    try:
        data = json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    except (ValueError, TypeError) as exc:
        raise ProtocolError(f'Cannot encode packet: {exc}') from exc
    if not 1 <= len(data) <= MAX_PACKET_BYTES:
        raise ProtocolError('Packet exceeds size limit')
    sock.sendall(struct.pack('!I', len(data)) + data)


def recv_packet(sock):
    length = struct.unpack('!I', _read_exact(sock, 4))[0]
    if not 1 <= length <= MAX_PACKET_BYTES:
        raise ProtocolError('Invalid packet length')
    try:
        result = json.loads(_read_exact(sock, length).decode('utf-8'),
                            parse_constant=_json_constant, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ProtocolError(f'Invalid JSON packet: {exc}') from exc
    if not isinstance(result, dict):
        raise ProtocolError('Packet must be an object')
    if result.get('version') != VERSION or type(result.get('version')) is not int:
        raise ProtocolError('Unsupported protocol version')
    return result


def encode_image(rgb, quality=95):
    import numpy as np
    from PIL import Image
    if not isinstance(rgb, np.ndarray) or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ProtocolError('Expected uint8 RGB image with shape (480,640,3)')
    if type(quality) is not int or not 80 <= quality <= 100:
        raise ProtocolError('JPEG quality must be an integer in [80,100]')
    output = io.BytesIO()
    Image.fromarray(rgb, mode='RGB').save(output, format='JPEG', quality=quality, subsampling=0)
    data = output.getvalue()
    if len(data) > MAX_JPEG_BYTES:
        raise ProtocolError('JPEG exceeds image size limit')
    return {'encoding': 'jpeg', 'width': 640, 'height': 480,
            'data': base64.b64encode(data).decode('ascii')}


def _jpeg_bytes(encoded):
    if (not isinstance(encoded, dict) or encoded.get('encoding') != 'jpeg'
            or encoded.get('width') != 640 or encoded.get('height') != 480
            or not isinstance(encoded.get('data'), str)):
        raise ProtocolError('Expected a 640x480 JPEG image envelope')
    if len(encoded['data']) > (MAX_JPEG_BYTES + 2) // 3 * 4:
        raise ProtocolError('JPEG exceeds image size limit')
    try:
        data = base64.b64decode(encoded['data'], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ProtocolError('Invalid base64 JPEG') from exc
    if not data or len(data) > MAX_JPEG_BYTES:
        raise ProtocolError('Invalid JPEG length')
    return data


def decode_image(encoded):
    import numpy as np
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(_jpeg_bytes(encoded))) as image:
            if image.size != (640, 480) or image.format != 'JPEG':
                raise ProtocolError('Decoded JPEG dimensions/format do not match training')
            return np.array(image.convert('RGB'), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ProtocolError(f'Invalid JPEG: {exc}') from exc


def validate_observation_request(packet):
    if packet.get('version') != VERSION or packet.get('type') != 'infer':
        raise ProtocolError('Expected a version 1 inference request')
    _session(packet.get('session_id'))
    _integer(packet.get('request_id'), 'request_id')
    _integer(packet.get('capture_seq'), 'capture_seq')
    if _number(packet.get('capture_monotonic'), 'capture_monotonic') < 0:
        raise ProtocolError('capture_monotonic must be nonnegative')
    obs = packet.get('observation')
    if not isinstance(obs, dict) or obs.get('task') != TASK:
        raise ProtocolError('Observation task does not match this avocado policy')
    _vector(obs.get('state'), 8, 'observation.state')
    images = obs.get('images')
    if not isinstance(images, dict) or set(images) != set(CAMERA_KEYS):
        raise ProtocolError('Observation must contain exactly the two training camera keys')
    for key in CAMERA_KEYS:
        _jpeg_bytes(images[key])
    return packet


def validate_metadata(packet):
    if packet.get('version') != VERSION or packet.get('type') != 'metadata':
        raise ProtocolError('Expected a version 1 metadata response')
    if packet.get('state_dim') != 8 or packet.get('action_dim') != 8:
        raise ProtocolError('Server must use 8D FR3 state and absolute actions')
    if packet.get('camera_keys') != list(CAMERA_KEYS) or packet.get('fps') != 15 or packet.get('chunk_size') != 50:
        raise ProtocolError('Server cameras, time step or action horizon do not match training')
    if packet.get('action_semantics') != ACTION_SEMANTICS:
        raise ProtocolError('Server action semantics are incompatible')
    if packet.get('task', TASK) != TASK:
        raise ProtocolError('Server task is incompatible')
    if type(packet.get('mock')) is not bool:
        raise ProtocolError('Server must explicitly identify real versus mock inference')
    if not isinstance(packet.get('model_id'), str) or not 1 <= len(packet['model_id']) <= 256:
        raise ProtocolError('Server must identify the loaded checkpoint')
    if not isinstance(packet.get('checkpoint'), str):
        raise ProtocolError('Server must report the pinned checkpoint path')
    return packet


def validate_actions_response(packet, request=None, model_id=None):
    if packet.get('version') != VERSION or packet.get('type') not in ('actions', 'submit'):
        raise ProtocolError('Expected version 1 actions')
    _session(packet.get('session_id'))
    for name in ('request_id', 'capture_seq'):
        _integer(packet.get(name), name)
    if _number(packet.get('capture_monotonic'), 'capture_monotonic') < 0:
        raise ProtocolError('capture_monotonic must be nonnegative')
    if packet.get('fps') != 15:
        raise ProtocolError('Actions must use the trained 15Hz time step')
    if not isinstance(packet.get('model_id'), str) or not packet['model_id']:
        raise ProtocolError('Actions lack model identity')
    if model_id is not None and packet['model_id'] != model_id:
        raise ProtocolError('Checkpoint changed during the session')
    if request is not None:
        for name in ('session_id', 'request_id', 'capture_seq', 'capture_monotonic'):
            if packet.get(name) != request.get(name):
                raise ProtocolError(f'Stale or mismatched action response: {name}')
    actions = packet.get('actions')
    if not isinstance(actions, list) or not 1 <= len(actions) <= 50:
        raise ProtocolError('Actions must contain 1 to 50 time steps')
    for row in actions:
        _vector(row, 8, 'action')
    if 'inference_s' in packet and _number(packet['inference_s'], 'inference_s') < 0:
        raise ProtocolError('Invalid inference duration')
    return packet
