#!/usr/bin/env python3
"""Offline converter CLI regressions using real small videos and collector schema 2.

All generated inputs, reports and attempted outputs stay in a temporary directory.
The deliberately broken shadow lerobot package proves --dry-run does not import it.
No GUI, camera, robot, network upload, training or deployment is involved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cv2
import h5py
import numpy as np

WORK = Path(tempfile.gettempdir())
DEFAULT_CONVERTER = Path(__file__).resolve().parents[1] / 'tools/convert_to_lerobot_v30.py'
JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]
FINGERS = ['fr3_finger_joint1', 'fr3_finger_joint2']
SERIALS = ['233522074047', '405622074065']
SOURCES = ['gello_joints', 'franka_joints', 'gripper_joints', 'gripper_command',
           'end_effector_pose', 'arm_command', 'gripper_target']
CONTRACT = 'fr3_joint_position_v1'
N = 12
FPS = 15
START = 1792548000.0


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def fixture_image(index: int, camera: int) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, camera] = 70 + index * 8
    cv2.putText(image, f'fixture camera {camera + 1} / frame {index}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def video(path: Path, count: int, camera: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (640, 480))
    assert writer.isOpened(), f'Local MP4 encoder unavailable: {path}'
    try:
        for index in range(count):
            writer.write(fixture_image(index, camera))
    finally:
        writer.release()
    check = cv2.VideoCapture(str(path))
    try:
        assert check.isOpened() and int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == count
    finally:
        check.release()


def make_episode(root: Path, name='record_001', outcome='success', schema=2, artifacts=True,
                 task='fixture: put the red block into the bowl') -> Path:
    record = root / name
    (record / 'action').mkdir(parents=True)
    (record / 'video').mkdir()
    # Match the real collector's ~31–139 ms intervals while retaining mean 15 Hz.
    jitter = np.asarray(([0.0, -0.036, 0.036, 0.0] * 3), dtype=np.float64)
    timestamps = START + np.arange(N) / FPS + jitter
    cameras = []
    for index, serial in enumerate(SERIALS, start=1):
        stem = f'camera_{index}_20261021_100000'
        cameras.append({'camera_index': index, 'serial': serial,
                        'video_path': f'video/{stem}.mp4',
                        'frames_directory': f'video/{stem}_frames'})
    metadata = {
        'schema_version': schema, 'capture_contract': CONTRACT if schema == 2 else None,
        'final_status_authority': 'metadata.json', 'session': name,
        'task': task, 'outcome': outcome,
        'started_at': '2026-10-21T10:00:00', 'color_resolution': [640, 480],
        'capture_fps': 30, 'record_fps': FPS, 'cameras_disabled': False, 'cameras': cameras,
        'status': 'complete', 'errors': [], 'frame_count': N, 'action_count': N,
        'observation_state_features': JOINTS + ['gripper_width_m'],
        'action_features': JOINTS + ['gripper_target_width_m'],
        'observation_state_units': ['rad'] * 7 + ['m'], 'action_units': ['rad'] * 7 + ['m'],
        'action_semantics': 'absolute_FR3_joint_position_targets_and_last_accepted_gripper_width',
        'clock_domain': 'ROS_system_epoch_seconds_and_nanoseconds',
        'causal_alignment': {'arm_max_age_s': 0.2, 'gripper_measurement_max_age_s': 0.5,
                             'gripper_heartbeat_max_age_s': 0.5,
                             'host_fallback_limitation': 'host receipt cannot prove exposure-time causality'},
        'image_time_bases': ['earliest_camera_exposure_system_epoch'],
        'controller': {'activation_id': str(round((START - 2) * 1e9)), 'relative_start': True,
                       'robot_start_q': [0.0] * 7, 'gello_start_q': [1.0] * 7,
                       'joint_names': JOINTS, 'sequence': 1, 'stamp_ns': round((START - 0.04) * 1e9)},
        'controller_config': {'source': 'fixture', 'parameters': {'k_gains': [120.0] * 7, 'd_gains': [10.0] * 7}},
        'gripper_contract': {'max_width_m': 0.08, 'observation': 'sum_of_two_measured_finger_positions_m',
                             'action': 'last_accepted_target_width_m', 'heartbeat_hz': 10},
        'gripper_control_profile': {'schema_version': 1, 'control_policy': 'move_width_v1',
                                    'policy_config': None},
    }
    dump(record / 'metadata.json', metadata)
    if not artifacts:
        return record
    frames, camera_times = [], []
    for index, timestamp in enumerate(timestamps):
        reference = float(timestamp - 0.03)
        arm_stamp = round((reference - 0.01) * 1e9)
        finger_stamp = round((reference - 0.02) * 1e9)
        heartbeat_stamp = round((reference - 0.04) * 1e9)
        measured = np.asarray([0.1, -0.2, 0.15, -1.4, 0.1, 1.5, 0.2], dtype=np.float64) + index * 0.001
        target = measured + 0.025
        fingers = [0.019 + index * 0.00001, 0.019 + index * 0.00001]
        arm = {'schema_version': 1, 'sequence': index * 3 + 1,
               'activation_id': metadata['controller']['activation_id'], 'stamp_ns': arm_stamp,
               'joint_names': JOINTS, 'q_measured': measured.tolist(), 'dq_measured': [0.015] * 7,
               'q_target': target.tolist(), 'tau_command': [0.2] * 7,
               'gello_q': (target + 1).tolist(), 'robot_start_q': [0.0] * 7,
               'gello_start_q': [1.0] * 7, 'relative_start': True}
        gripper_target = {'schema_version': 1, 'sequence': 1, 'accepted_sequence': 1,
                          'stamp_ns': heartbeat_stamp, 'command_stamp_ns': round((START - 1) * 1e9),
                          'target_width_m': 0.04, 'speed_m_s': 0.05, 'max_width_m': 0.08,
                          'valid': True, 'status': 'completed', 'requested_width_m': 0.04,
                          'requested_speed_m_s': 0.05, 'error': ''}
        source_times = {}
        for key in SOURCES:
            stamp = finger_stamp if key == 'gripper_joints' else heartbeat_stamp if key == 'gripper_target' else arm_stamp
            source_times[key] = {'source_stamp_ns': None if key == 'gripper_command' else stamp,
                                 'host_received_ts': reference - 0.001, 'topic': '/fixture/' + key}
        ages = {key: (round(reference * 1e9) - source_times[key]['source_stamp_ns']) / 1e9
                for key in ('arm_command', 'gripper_joints', 'gripper_target')}
        frames.append({
            'timestamp': arm_stamp / 1e9, 'datetime': '2026-10-21T02:00:00',
            'bridge_packet_timestamp': float(timestamp - 0.002), 'frame_timestamp': float(timestamp),
            'gello_joints': {'names': JOINTS, 'position': arm['gello_q'], 'velocity': None, 'effort': None},
            'franka_joints': {'names': JOINTS, 'position': arm['q_measured'], 'velocity': arm['dq_measured'], 'effort': [0.1] * 7},
            'gripper_joints': {'names': FINGERS, 'source_names': FINGERS, 'position': fingers, 'velocity': [0.0, 0.0], 'effort': [0.0, 0.0]},
            'gripper_command': 0.5, 'gripper_target': gripper_target, 'arm_command': arm,
            'end_effector_pose': {'position': {'x': 0.4, 'y': 0.0, 'z': 0.3}, 'orientation': {'x': 1.0, 'y': 0.0, 'z': 0.0, 'w': 0.0}},
            'source_timestamps': source_times,
            'observation.state': np.asarray(measured.tolist() + [sum(fingers)], dtype=np.float32).tolist(),
            'action': np.asarray(target.tolist() + [0.04], dtype=np.float32).tolist(),
            'alignment': {'reference_timestamp': reference, 'source_age_s': ages,
                          'cross_stream_skew_s': max(ages.values()) - min(ages.values()),
                          'selection': 'latest_source_sample_at_or_before_image_reference',
                          'image_time_basis': 'earliest_camera_exposure_system_epoch',
                          'image_host_timestamp': float(timestamp)},
        })
        camera_times.append({'frame_index': index, 'frame_timestamp': float(timestamp), 'cameras': [
            {'serial': serial, 'host_timestamp': float(timestamp - 0.0002 + camera * 0.0001),
             'device_timestamp_ms': (reference + camera * 0.005) * 1000,
             'device_frame_number': index * 2 + 100 + camera,
             'timestamp_domain': 'timestamp_domain.global_time'} for camera, serial in enumerate(SERIALS)]})
    payload = {'schema_version': 2, 'capture_contract': CONTRACT, 'session': name,
               'timestamp': '20261021_100001_000000', 'task': metadata['task'], 'outcome': outcome,
               'save_status': 'complete', 'packet_count': N, 'metadata': copy.deepcopy(metadata), 'data': frames}
    dump(record / 'action/action_data_20261021_100001_000000.json', payload)
    dump(record / 'video/frame_timestamps.json', camera_times)
    with h5py.File(record / 'action/action_data_20261021_100001_000000.h5', 'w') as handle:
        for key, value in {'schema_version': 2, 'capture_contract': CONTRACT, 'session': name,
                           'timestamp': '20261021_100001_000000', 'packet_count': N,
                           'task': metadata['task'], 'outcome': outcome, 'record_fps': FPS,
                           'save_status': 'complete', 'metadata_json': json.dumps(metadata),
                           'observation_state_features': json.dumps(metadata['observation_state_features']),
                           'action_features': json.dumps(metadata['action_features']),
                           'feature_units': json.dumps(['rad'] * 7 + ['m']),
                           'action_semantics': 'absolute_joint_position_targets_and_last_accepted_gripper_width'}.items():
            handle.attrs[key] = value
        handle.create_dataset('timestamps', data=timestamps)
        handle.create_dataset('observation/state', data=np.asarray([frame['observation.state'] for frame in frames], dtype=np.float32))
        handle.create_dataset('action', data=np.asarray([frame['action'] for frame in frames], dtype=np.float32))
        for key in ('q_target', 'tau_command', 'q_measured', 'dq_measured', 'gello_q', 'robot_start_q', 'gello_start_q'):
            handle.create_dataset('control/' + key, data=np.asarray([frame['arm_command'][key] for frame in frames], dtype=np.float32))
        for key in ('sequence', 'stamp_ns'):
            handle.create_dataset('control/' + key, data=np.asarray([frame['arm_command'][key] for frame in frames], dtype=np.int64))
        handle.create_dataset('control/activation_id', data=np.asarray([metadata['controller']['activation_id']] * N, dtype=object), dtype=h5py.string_dtype('utf-8'))
        handle.create_dataset('control/relative_start', data=np.ones(N, dtype=np.bool_))
        for source, prefix, size in [('gello_joints', 'gello', 7), ('franka_joints', 'franka', 7), ('gripper_joints', 'gripper', 2)]:
            for key, suffix in [('position', 'positions'), ('velocity', 'velocities'), ('effort', 'efforts')]:
                handle.create_dataset(prefix + '/' + suffix, data=np.asarray([
                    frame[source][key] if frame[source][key] is not None else [np.nan] * size for frame in frames], dtype=np.float32))
        handle.create_dataset('gripper/width_m', data=np.asarray([sum(frame['gripper_joints']['position']) for frame in frames], dtype=np.float32))
        for key in ('target_width_m', 'max_width_m', 'speed_m_s'):
            handle.create_dataset('gripper/' + key, data=np.asarray([frame['gripper_target'][key] for frame in frames], dtype=np.float32))
        for key in ('stamp_ns', 'command_stamp_ns', 'sequence', 'accepted_sequence'):
            handle.create_dataset('gripper/target_' + key, data=np.asarray([frame['gripper_target'][key] for frame in frames], dtype=np.int64))
        handle.create_dataset('gripper/target_status', data=np.asarray(['completed'] * N, dtype=object), dtype=h5py.string_dtype('utf-8'))
        handle.create_dataset('gripper/raw_command_percent', data=np.full(N, 0.5, dtype=np.float32))
        handle.create_dataset('end_effector/positions', data=np.tile([0.4, 0.0, 0.3], (N, 1)).astype(np.float32))
        handle.create_dataset('end_effector/orientations', data=np.tile([1.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32))
        handle.create_dataset('alignment/reference_timestamp', data=np.asarray([frame['alignment']['reference_timestamp'] for frame in frames]))
        handle.create_dataset('alignment/cross_stream_skew_s', data=np.asarray([frame['alignment']['cross_stream_skew_s'] for frame in frames], dtype=np.float32))
        handle.create_dataset('alignment/image_time_basis', data=np.asarray(['earliest_camera_exposure_system_epoch'] * N, dtype=object), dtype=h5py.string_dtype('utf-8'))
        for key in SOURCES:
            handle.create_dataset('source_timestamps/' + key + '/source_stamp_ns', data=np.asarray([
                frame['source_timestamps'][key]['source_stamp_ns'] if frame['source_timestamps'][key]['source_stamp_ns'] is not None else -1 for frame in frames], dtype=np.int64))
            handle.create_dataset('source_timestamps/' + key + '/host_received_ts', data=np.asarray([frame['source_timestamps'][key]['host_received_ts'] for frame in frames]))
        for key in ('arm_command', 'gripper_joints', 'gripper_target'):
            handle.create_dataset('alignment/source_age_s/' + key, data=np.asarray([frame['alignment']['source_age_s'][key] for frame in frames], dtype=np.float32))
    for index, camera in enumerate(cameras):
        video(record / camera['video_path'], N, index)
        raw_dir = record / camera['frames_directory']
        raw_dir.mkdir()
        for frame in range(N):
            assert cv2.imwrite(str(raw_dir / f'frame_{frame:06d}.jpg'), fixture_image(frame, index))
    return record


def fingerprint(root: Path) -> dict:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


class Suite:
    def __init__(self, converter: Path, root: Path):
        self.converter, self.root, self.count = converter, root, 0
        self.env = os.environ.copy()
        shadow = root / 'shadow_packages/lerobot'
        shadow.mkdir(parents=True)
        (shadow / '__init__.py').write_text("raise RuntimeError('DRY_RUN_MUST_NOT_IMPORT_LEROBOT')\n")
        self.env['PYTHONPATH'] = str(shadow.parent) + os.pathsep + self.env.get('PYTHONPATH', '')
        self.env['HF_HUB_OFFLINE'] = '1'
        self.env['HF_DATASETS_OFFLINE'] = '1'
        self.base = root / 'base/input'
        make_episode(self.base)
        make_episode(self.base, 'record_002', schema=1, artifacts=False)
        make_episode(self.base, 'record_003', outcome='unknown', artifacts=False)
        make_episode(self.base, 'record_004', outcome='failure', artifacts=False)
        incomplete = make_episode(self.base, 'record_005', artifacts=False)
        incomplete_meta = json.loads((incomplete / 'metadata.json').read_text())
        incomplete_meta['status'] = 'recording'
        dump(incomplete / 'metadata.json', incomplete_meta)

    def invoke(self, name, input_dir=None, extra=(), expected=0, existing_output=False,
               report_arg=None, expect_report=True):
        case = self.root / name
        case.mkdir(parents=True, exist_ok=True)
        input_dir = input_dir or self.base
        output = case / 'output'
        report = case / 'report.json'
        if existing_output:
            output.mkdir()
            (output / 'sentinel.txt').write_text('must remain unchanged')
        before = fingerprint(input_dir)
        command = [sys.executable, str(self.converter), '--input-dir', str(input_dir),
                   '--output-dir', str(output), '--repo-id', 'local/test', '--dry-run',
                   '--report', str(report_arg) if report_arg is not None else str(report), *extra]
        result = subprocess.run(command, cwd=case, env=self.env, capture_output=True, text=True, timeout=120)
        assert 'DRY_RUN_MUST_NOT_IMPORT_LEROBOT' not in result.stdout + result.stderr, result.stderr
        if expected == 0:
            assert result.returncode == 0, f'{name}: exit {result.returncode}\n{result.stdout}\n{result.stderr}'
        else:
            assert result.returncode != 0, f'{name}: malformed selected success silently passed\n{result.stdout}'
        assert before == fingerprint(input_dir), f'{name}: converter changed source data'
        if existing_output:
            assert (output / 'sentinel.txt').read_text() == 'must remain unchanged'
            assert sorted(path.name for path in output.iterdir()) == ['sentinel.txt']
        else:
            assert not output.exists(), f'{name}: --dry-run created output directory'
        self.count += 1
        if not expect_report:
            return None
        if report.exists():
            return json.loads(report.read_text())
        assert existing_output, f'{name}: requested JSON report was not written\n{result.stdout}\n{result.stderr}'
        return None

    def clone(self, name):
        dest = self.root / name / 'input'
        shutil.copytree(self.base, dest)
        return dest

    def invalid(self, name, mutate):
        source = self.clone(name)
        mutate(source / 'record_001')
        report = self.invoke(name, source, expected=1)
        assert report.get('errors') and any(error.get('record') == 'record_001' and error.get('error') for error in report['errors']), report
        print('PASS:', name)

    def run(self):
        report = self.invoke('valid_jitter_and_filters')
        assert [item['record'] for item in report['selected']] == ['record_001'], report
        assert report['selected'][0]['frames'] == N and report['total_frames'] == N and report['fps'] == FPS
        assert {item['record'] for item in report['skipped']} == {'record_002', 'record_003', 'record_004', 'record_005'}
        assert all(item.get('reason') for item in report['skipped']) and not report.get('errors')
        print('PASS: real two-camera MP4, legal jitter, complete+success filtering, explicit skip reasons, no lerobot import or output writes')
        self.invoke('existing_output', existing_output=True, expected=1)
        print('PASS: existing output refused without changing its sentinel')

        relative_input = self.clone('report_inside_input_relative_earlyerror')
        self.invoke('report_inside_input_relative_earlyerror', relative_input, existing_output=True,
                    expected=1, report_arg='input/record_001/metadata.json', expect_report=False)
        print('PASS: early output error cannot overwrite an input metadata file through a relative report path')

        notes_case = self.root / 'report_existing_user_file'
        notes_case.mkdir()
        notes = notes_case / 'notes.json'
        notes.write_text('{"project": "human notes", "do_not_overwrite": true}\n')
        original_notes = notes.read_bytes()
        self.invoke('report_existing_user_file', expected=1, report_arg='notes.json', expect_report=False)
        assert notes.read_bytes() == original_notes
        print('PASS: report refuses to overwrite an existing non-converter JSON file')

        def bad_action(record):
            path = next((record / 'action').glob('*.json'))
            doc = json.loads(path.read_text()); doc['data'][3]['action'] = doc['data'][3]['action'][:7]; dump(path, doc)
        self.invalid('malformed_success_action_shape', bad_action)

        def wrong_action_semantics(record):
            path = next((record / 'action').glob('*.json'))
            doc = json.loads(path.read_text())
            for frame in doc['data']:
                frame['action'][:7] = np.asarray(frame['arm_command']['gello_q'], dtype=np.float32).tolist()
            dump(path, doc)
            with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle:
                handle['action'][:] = np.asarray([frame['action'] for frame in doc['data']], dtype=np.float32)
        self.invalid('malformed_success_gello_substituted_for_target', wrong_action_semantics)

        def h5_mismatch(record):
            with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle:
                handle['action'][3, 0] += 0.1
        self.invalid('malformed_success_h5_json_mismatch', h5_mismatch)

        def h5_epoch_time_offset(record):
            with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle:
                handle['timestamps'][:] = handle['timestamps'][:] + 0.04
        self.invalid('malformed_success_h5_epoch_time_40ms_offset', h5_epoch_time_offset)

        def accepted_command_changed(record):
            path = next((record / 'action').glob('*.json'))
            doc = json.loads(path.read_text())
            doc['data'][4]['gripper_target']['command_stamp_ns'] += 1_000_000
            dump(path, doc)
            with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle:
                handle['gripper/target_command_stamp_ns'][4] = doc['data'][4]['gripper_target']['command_stamp_ns']
        self.invalid('malformed_success_same_accepted_id_changed_command_time', accepted_command_changed)

        def accepted_width_changed(record):
            path = next((record / 'action').glob('*.json'))
            doc = json.loads(path.read_text())
            doc['data'][4]['gripper_target']['target_width_m'] = 0.045
            doc['data'][4]['gripper_target']['requested_width_m'] = 0.045
            doc['data'][4]['action'][7] = float(np.float32(0.045))
            dump(path, doc)
            with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle:
                handle['gripper/target_width_m'][4] = 0.045
                handle['action'][4, 7] = 0.045
        self.invalid('malformed_success_same_accepted_id_changed_width', accepted_width_changed)

        def duplicate_arm_snapshot(record, change_target=False):
            path = next((record / 'action').glob('*.json'))
            doc = json.loads(path.read_text())
            previous, frame = doc['data'][3], doc['data'][4]
            arm = copy.deepcopy(previous['arm_command'])
            if change_target:
                arm['q_target'] = (np.asarray(arm['q_target']) + 0.01).tolist()
                arm['gello_q'] = (np.asarray(arm['q_target']) + 1).tolist()
            frame['arm_command'] = arm
            frame['source_timestamps']['arm_command'] = copy.deepcopy(previous['source_timestamps']['arm_command'])
            frame['timestamp'] = arm['stamp_ns'] / 1e9
            frame['observation.state'][:7] = np.asarray(arm['q_measured'], dtype=np.float32).tolist()
            frame['action'][:7] = np.asarray(arm['q_target'], dtype=np.float32).tolist()
            ages = frame['alignment']['source_age_s']
            ages['arm_command'] = (round(frame['alignment']['reference_timestamp'] * 1e9) - arm['stamp_ns']) / 1e9
            frame['alignment']['cross_stream_skew_s'] = max(ages.values()) - min(ages.values())
            dump(path, doc)
            with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle:
                for key in ('q_target', 'tau_command', 'q_measured', 'dq_measured', 'gello_q', 'robot_start_q', 'gello_start_q', 'sequence', 'stamp_ns'):
                    handle['control/' + key][4] = arm[key]
                for key in ('source_stamp_ns', 'host_received_ts'):
                    handle['source_timestamps/arm_command/' + key][4] = frame['source_timestamps']['arm_command'][key]
                handle['observation/state'][4] = frame['observation.state']
                handle['action'][4] = frame['action']
                handle['alignment/source_age_s/arm_command'][4] = ages['arm_command']
                handle['alignment/cross_stream_skew_s'][4] = frame['alignment']['cross_stream_skew_s']

        identical = self.clone('duplicate_identical_arm_snapshot_allowed')
        duplicate_arm_snapshot(identical / 'record_001')
        identical_report = self.invoke('duplicate_identical_arm_snapshot_allowed', identical)
        assert identical_report['total_frames'] == N and not identical_report.get('errors')
        print('PASS: repeated unchanged arm sequence/stamp remains a valid cached snapshot')
        self.invalid('malformed_success_same_arm_identity_changed_target', lambda record: duplicate_arm_snapshot(record, True))

        def missing_h5_control(record):
            with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle:
                del handle['control/q_target']
        self.invalid('malformed_success_missing_control', missing_h5_control)

        def nonmonotonic(record):
            path = next((record / 'action').glob('*.json'))
            doc = json.loads(path.read_text()); bad = doc['data'][5]['frame_timestamp'] - 0.001
            doc['data'][6]['frame_timestamp'] = bad; dump(path, doc)
            with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle: handle['timestamps'][6] = bad
            sidecar = record / 'video/frame_timestamps.json'; frames = json.loads(sidecar.read_text()); frames[6]['frame_timestamp'] = bad; dump(sidecar, frames)
        self.invalid('malformed_success_nonmonotonic_time', nonmonotonic)

        def short_video(record):
            meta = json.loads((record / 'metadata.json').read_text())
            video(record / meta['cameras'][1]['video_path'], N - 1, 1)
        self.invalid('malformed_success_missing_video_frame', short_video)

        def missing_task(record):
            meta = json.loads((record / 'metadata.json').read_text()); meta['task'] = ''; dump(record / 'metadata.json', meta)
        self.invalid('malformed_success_empty_task', missing_task)

        host_input = self.clone('host_timestamp_fallback')
        record = host_input / 'record_001'
        meta = json.loads((record / 'metadata.json').read_text()); meta['image_time_bases'] = ['host_received']; dump(record / 'metadata.json', meta)
        path = next((record / 'action').glob('*.json')); doc = json.loads(path.read_text())
        for frame in doc['data']:
            frame['alignment']['image_time_basis'] = 'host_received'
            frame['alignment']['reference_timestamp'] = frame['frame_timestamp']
            frame['alignment']['source_age_s'] = {key: frame['frame_timestamp'] - frame['source_timestamps'][key]['source_stamp_ns'] / 1e9
                                                   for key in ('arm_command', 'gripper_joints', 'gripper_target')}
        doc['metadata'] = meta; dump(path, doc)
        sidecar = record / 'video/frame_timestamps.json'; camera_frames = json.loads(sidecar.read_text())
        for index, frame in enumerate(camera_frames):
            for camera in frame['cameras']:
                camera['timestamp_domain'] = 'timestamp_domain.hardware_clock'
                camera['device_timestamp_ms'] = 5000.0 + index * 1000 / FPS
        dump(sidecar, camera_frames)
        with h5py.File(next((record / 'action').glob('*.h5')), 'r+') as handle:
            handle.attrs['metadata_json'] = json.dumps(meta)
            handle['alignment/image_time_basis'][:] = ['host_received'] * N
            handle['alignment/reference_timestamp'][:] = [frame['frame_timestamp'] for frame in doc['data']]
            for key in ('arm_command', 'gripper_joints', 'gripper_target'):
                handle['alignment/source_age_s/' + key][:] = [frame['alignment']['source_age_s'][key] for frame in doc['data']]
        rejected = self.invoke('host_timestamp_fallback_default', host_input, expected=1)
        assert rejected.get('errors'), rejected
        allowed = self.invoke('host_timestamp_fallback_opt_in', host_input, extra=['--allow-host-timestamps'])
        assert len(allowed['selected']) == 1 and allowed['total_frames'] == N
        print('PASS: host timestamp fallback rejected by default and accepted only with explicit opt-in')
        print(f'All {self.count} offline converter CLI cases passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--converter', type=Path, default=DEFAULT_CONVERTER)
    parser.add_argument('--keep-fixtures', action='store_true')
    args = parser.parse_args()
    assert args.converter.is_file(), f'Converter is not available yet: {args.converter}'
    root = Path(tempfile.mkdtemp(prefix='convert_v30_test_', dir=WORK))
    succeeded = False
    try:
        Suite(args.converter.resolve(), root).run()
        succeeded = True
    finally:
        if succeeded and not args.keep_fixtures:
            shutil.rmtree(root)
        else:
            print('Fixtures/reports retained at:', root, file=sys.stderr)


if __name__ == '__main__':
    main()
