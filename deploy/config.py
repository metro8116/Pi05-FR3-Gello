"""Deployment settings, shared by host and ROS Python; no environment changes."""
from __future__ import annotations
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_CONFIG = ROOT / 'configs/deploy.yaml'


def load_config(path=None):
    """The shipped .yaml uses JSON syntax (valid YAML), needing only stdlib."""
    path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'{path}: use JSON syntax as in the supplied YAML file: {exc}') from exc
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('Unsupported deployment config schema_version')
    for name in ('model', 'bridge', 'cameras', 'execution', 'robot'):
        if not isinstance(config.get(name), dict):
            raise ValueError(f'Missing configuration section: {name}')
    if config.get('task') != 'Put the avocado in the basket.':
        raise ValueError('This deployment is configured for the avocado task')
    for name in ('model','bridge'):
        if config[name].get('host') != '127.0.0.1':
            raise ValueError(f'{name}.host must be 127.0.0.1; use an SSH tunnel for the remote server')
        port = config[name].get('port')
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError(f'Invalid {name}.port')
    c = config['cameras']
    if (c.get('serials') != ['233522074047','405622074065'] or c.get('width') != 640
            or c.get('height') != 480 or c.get('capture_fps') != 30):
        raise ValueError('Camera order/resolution must match the training capture')
    if config['execution'].get('fps') != 15:
        raise ValueError('The trained action time step is 1/15 second')
    recording = config.setdefault('recording', {'directory': 'recordings', 'queue_size': 64})
    if (not isinstance(recording, dict) or not isinstance(recording.get('directory'), str)
            or not recording['directory'].strip() or type(recording.get('queue_size')) is not int
            or not 2 <= recording['queue_size'] <= 256):
        raise ValueError('recording requires a directory and a bounded queue_size in 2..256')
    return copy.deepcopy(config)
