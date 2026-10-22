"""Mock source telemetry and verify the capture contract; never access hardware."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import sys

SOURCE = str(Path(__file__).resolve().parents[1] / 'capture/video_action_recorder.py')
try:
    for dependency in ("cv2", "pyrealsense2", "tkinter", "h5py", "numpy", "PIL"):
        importlib.import_module(dependency)
except ImportError as exc:
    reason = "Optional capture dependencies unavailable: " + str(exc)
    if __name__ == "__main__":
        print("SKIP:", reason)
        raise SystemExit(0)
    raise unittest.SkipTest(reason) from exc

spec = importlib.util.spec_from_file_location('capture_contract_test', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Widget:
    def __init__(self): self.values = {}
    def config(self, **kwargs): self.values.update(kwargs)
    def set(self, value): self.value = value


class Root:
    def after(self, delay, callback, *args):
        if delay == 0: callback(*args)


class HeadlessRecorder(mod.VideoActionRecorder):
    def _acquire_fr3_control_lease(self): pass
    def create_gui(self):
        self.root = Root()
        for name in ('camera_status_label', 'preview_label', 'robot_ready_label', 'robot_connection_label',
                     'session_label', 'status_label', 'start_button', 'stop_button', 'delete_last_button',
                     'time_label', 'video_path_label', 'action_path_label', 'replay_index_var', 'teleop_button'):
            setattr(self, name, Widget())
    def load_ui_state(self): return {}
    def setup_cleanup_hooks(self): pass
    def _update_replay_controls(self): pass
    def _capture_controller_config(self): return {'source': 'mock_configuration'}


def packet(arm_time=99.95, finger_time=99.9, heartbeat=99.9, activation='1000', measured=0.12, target=0.6):
    arm = {'schema_version': 1, 'sequence': 17, 'activation_id': activation,
           'stamp_ns': round(arm_time * 1e9), 'joint_names': mod.FR3_JOINT_ORDER,
           'q_measured': [measured] * 7, 'dq_measured': [0.01] * 7,
           'q_target': [target] * 7, 'tau_command': [1.2] * 7,
           'gello_q': [2.4] * 7, 'robot_start_q': [-0.2] * 7,
           'gello_start_q': [1.6] * 7, 'relative_start': True}
    goal = {'schema_version': 1, 'sequence': 8, 'accepted_sequence': 7,
            'stamp_ns': round(heartbeat * 1e9), 'command_stamp_ns': round(80 * 1e9),
            'target_width_m': 0.07, 'speed_m_s': 0.05, 'max_width_m': 0.08,
            'valid': True, 'status': 'accepted'}
    data = {'timestamp': 100.0, 'datetime': 'fixture', 'arm_command': arm, 'gripper_target': goal,
            'gripper_command': 0.9,
            'gripper_joints': {'names': mod.FINGER_JOINT_ORDER, 'position': [0.02, 0.03], 'velocity': None, 'effort': None},
            'gello_joints': {'names': mod.FR3_JOINT_ORDER, 'position': [2.4] * 7},
            'franka_joints': {'names': mod.FR3_JOINT_ORDER, 'position': [9.0] * 7},
            'end_effector_pose': {'position': {'x': 0.3, 'y': 0.2, 'z': 0.4}, 'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}}}
    data['source_timestamps'] = {}
    for key in mod.CAPTURE_SOURCE_KEYS:
        stamp = finger_time if key == 'gripper_joints' else heartbeat if key == 'gripper_target' else arm_time
        data['source_timestamps'][key] = {'source_stamp_ns': None if key == 'gripper_command' else round(stamp * 1e9),
                                         'host_received_ts': 100.0, 'topic': '/fixture/' + key}
    return {'data': data, 'local_recv_ts': 100.0, 'remote_ts': 100.0}


def expect_error(app, sample, text):
    app.action_sample_buffer.clear()
    app.action_sample_buffer.append(sample)
    try:
        app._select_capture_sample(100.0)
    except (RuntimeError, ValueError, TypeError) as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError('Bad telemetry unexpectedly passed: ' + text)


errors, notices = [], []
with tempfile.TemporaryDirectory(prefix='pi05_capture_stub_', dir=str(Path(__file__).parent)) as directory, \
     patch.dict(os.environ, {'TAKE_DATA_DISABLE_CAMERAS': '1', 'TAKE_DATA_AUTOSTART_ROBOT': '0',
                             'TAKE_DATA_TASK': '', 'TAKE_DATA_OUTPUT_DIR': directory}), \
     patch.object(mod.messagebox, 'showerror', lambda title, message: errors.append(message)), \
     patch.object(mod.messagebox, 'showinfo', lambda title, message: notices.append(message)), \
     patch.object(mod.time, 'time', lambda: 100.0):
    with patch.dict(os.environ):
        os.environ.pop('TAKE_DATA_TASK', None)
        defaults = HeadlessRecorder()
        assert defaults.task_text == 'Put the avocado in the basket.'
        assert defaults.outcome == 'success'
    with patch.dict(os.environ, {'TAKE_DATA_TASK': 'A manually supplied task'}):
        customized = HeadlessRecorder()
        assert customized.task_text == 'A manually supplied task'
    print('PASS: requested avocado task/success defaults and explicit custom task are preserved')
    app = HeadlessRecorder()
    assert app.task_text == ''  # An explicitly empty task still exercises the refusal gate.
    app.action_sample_buffer.append(packet())
    assert not app._start_recording_core()
    assert not list(Path(directory).glob('record_*'))
    app.task_text = '把红色积木放入盒子'

    # Canonical state/action must use controller telemetry and measured fingers.
    sample = app._select_capture_sample(100.0)
    mod.np.testing.assert_allclose(sample['observation.state'], [0.12] * 7 + [0.05])
    mod.np.testing.assert_allclose(sample['action'], [0.6] * 7 + [0.07])
    assert sample['action'][:7] != sample['gello_joints']['position']
    assert sample['observation.state'][:7] != sample['franka_joints']['position']
    print('PASS: empty tasks refused; observation is measured arm+fingers, action is actual controller+accepted gripper target')

    # Future state and even a nearer post-exposure state cannot enter observations.
    app.action_sample_buffer.extend([packet(arm_time=99.99, measured=0.8, target=0.9),
                                     packet(arm_time=100.01, measured=1.8, target=1.9)])
    app._capture_action_sample(100.0, [
        {'timestamp_domain': 'timestamp_domain.global_time', 'device_timestamp_ms': 99970.0},
        {'timestamp_domain': 'timestamp_domain.system_time', 'device_timestamp_ms': 99980.0}])
    assert app.action_data_buffer[-1]['alignment']['reference_timestamp'] == 99.97
    assert app.action_data_buffer[-1]['alignment']['image_time_basis'] == 'earliest_camera_exposure_system_epoch'
    assert abs(app.action_data_buffer[-1]['observation.state'][0] - 0.12) < 1e-6
    assert all(age >= 0 for age in app.action_data_buffer[-1]['alignment']['source_age_s'].values())
    print('PASS: selection uses the earliest camera exposure and excludes future arm states')

    # Repeated TCP packets never rejuvenate old source timestamps.
    expect_error(app, packet(arm_time=99.7), 'arm_command 源数据已过期')
    expect_error(app, packet(finger_time=99.4), 'gripper_joints 源数据已过期')
    expect_error(app, packet(heartbeat=99.4), 'gripper_target 源数据已过期')
    expect_error(app, packet(arm_time=100.1), '缺少图像时刻之前的 arm_command')
    missing = packet(); missing['data']['gripper_joints'] = None
    expect_error(app, missing, 'gripper_joints')
    bad = packet(); bad['data']['gripper_target']['valid'] = False
    expect_error(app, bad, '夹爪最近接受的目标无效')
    bad = packet(); bad['data']['gripper_target']['status'] = 'error'
    expect_error(app, bad, '状态错误')
    bad = packet(); bad['data']['gripper_target']['command_stamp_ns'] = round(100.01 * 1e9)
    expect_error(app, bad, '缺少图像之前已接受的夹爪命令')
    pending = packet(); pending['data']['gripper_target']['status'] = 'pending'
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(pending)
    assert app._select_capture_sample(100.0)['action'][7] == mod.np.float32(0.07)
    print('PASS: source-age, future, missing-measurement, invalid/error target gates; accepted target survives pending request')

    # A normal unsuccessful Move is a target outcome, not a recording fault or task label.
    not_reached = packet()
    not_reached['data']['gripper_target'].update(status='not_reached', result_sequence=7,
                                               result_status=6, result_success=False,
                                               result_error='', error=None)
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(not_reached)
    unchanged_outcome = app.outcome
    sample = app._select_capture_sample(100.0)
    assert sample['gripper_target']['result_success'] is False and app.outcome == unchanged_outcome
    mod.np.testing.assert_allclose(sample['observation.state'], [0.12] * 7 + [0.05])
    mod.np.testing.assert_allclose(sample['action'], [0.6] * 7 + [0.07])
    assert sample['gripper_target']['accepted_sequence'] == 7
    assert sample['gripper_target']['command_stamp_ns'] == round(80 * 1e9)
    for updates, text in (({'result_sequence': 6}, 'not_reached'),
                          ({'result_sequence': False}, 'result_sequence'),
                          ({'result_status': 4}, 'not_reached'),
                          ({'result_status': 5}, 'not_reached'),
                          ({'result_status': 6.0}, 'result_status'),
                          ({'result_success': True}, 'not_reached'),
                          ({'result_success': 0}, 'result_success'),
                          ({'result_error': 'libfranka exception'}, 'not_reached'),
                          ({'result_error': None}, 'not_reached'),
                          ({'error': 'transport exception'}, '状态错误'),
                          ({'valid': 1}, '目标无效'),
                          ({'accepted_sequence': True}, 'not_reached')):
        bad = copy.deepcopy(not_reached)
        bad['data']['gripper_target'].update(updates)
        expect_error(app, bad, text)
    for field in ('result_sequence', 'result_status', 'result_success', 'result_error'):
        bad = copy.deepcopy(not_reached)
        del bad['data']['gripper_target'][field]
        expect_error(app, bad, 'not_reached')
    bad = copy.deepcopy(not_reached)
    bad['data']['gripper_target'].update(status='error', result_error='real error', error='real error')
    expect_error(app, bad, '状态错误')
    assert app._last_capture_diagnostics['latest_gripper_target'] == bad['data']['gripper_target']
    bad['data']['gripper_target']['result_error'] = 'changed after snapshot'
    assert app._last_capture_diagnostics['latest_gripper_target']['result_error'] == 'real error'
    print('PASS: narrowly classified not_reached keeps measured state, accepted action and task label; malformed/exception/canceled results rejected; raw failure diagnostics copied')

    # A new controller activation ends the current episode instead of merging zeros.
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(packet(activation='2000'))
    try:
        app._capture_action_sample(100.0)
    except RuntimeError as exc:
        assert 'activation_id' in str(exc)
    else: raise AssertionError('Cross-activation sample was accepted')
    app.action_data_buffer.clear()
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(packet())
    assert app._start_recording_core()
    app._capture_action_sample(100.0)
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(copy.deepcopy(not_reached))
    app._capture_action_sample(100.0)
    completed = packet()
    completed['data']['gripper_target'].update(status='completed', result_sequence=7,
                                             result_status=4, result_success=True,
                                             result_error='', error=None)
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(completed)
    app._capture_action_sample(100.0)
    app.outcome = 'failure'
    app.stop_recording()
    record = Path(app.session_dir['record'])
    metadata = json.loads((record / 'metadata.json').read_text())
    assert metadata['schema_version'] == 2 and metadata['capture_contract'] == 'fr3_joint_position_v1'
    assert metadata['task'] == app.task_text and metadata['outcome'] == 'failure'
    assert metadata['controller']['activation_id'] == '1000'
    doc = json.loads(next((record / 'action').glob('*.json')).read_text())
    assert doc['task'] == app.task_text and doc['outcome'] == 'failure'
    assert 'arm_command' in doc['data'][0] and 'gripper_command' in doc['data'][0]
    assert 'result_success' not in doc['data'][0]['gripper_target']
    assert doc['data'][1]['gripper_target']['result_success'] is False
    assert doc['data'][2]['gripper_target']['result_success'] is True
    with mod.h5py.File(next((record / 'action').glob('*.h5')), 'r') as handle:
        assert handle['observation/state'].shape == (3, 8) and handle['observation/state'].dtype == 'float32'
        assert handle['action'].shape == (3, 8) and handle['action'].dtype == 'float32'
        assert handle['gripper/positions'].shape == (3, 2)
        mod.np.testing.assert_allclose(handle['gripper/width_m'][:], [0.05] * 3)
        assert handle['gripper/target_result_success'].dtype == 'int8'
        assert list(handle['gripper/target_result_success'][:]) == [-1, 0, 1]
        assert list(handle['gripper/target_result_status'][:]) == [-1, 6, 4]
        assert list(handle['gripper/target_result_sequence'][:]) == [-1, 7, 7]
        assert list(handle['gripper/target_result_error'].asstr()[:]) == ['', '', '']
        assert list(handle['gripper/target_status'].asstr()[:]) == ['accepted', 'not_reached', 'completed']
        assert '-1=no_result' in handle['gripper/target_result_success'].attrs['encoding']
        mod.np.testing.assert_allclose(handle['action'][:, :7], 0.6)
        mod.np.testing.assert_allclose(handle['control/q_target'][:], 0.6)
        mod.np.testing.assert_allclose(handle['control/tau_command'][:], 1.2)
        assert handle.attrs['task'] == app.task_text and handle.attrs['outcome'] == 'failure'
        assert list(handle['control/activation_id'].asstr()[:]) == ['1000'] * 3
        assert handle['source_timestamps/arm_command/source_stamp_ns'][0] == 99950000000
        assert handle['alignment/source_age_s/arm_command'][0] > 0
    print('PASS: episode activation boundary, schema-2 task/outcome, raw telemetry and canonical float32 JSON/HDF5 write/readback')

    # A quality error stops only recording, saves valid earlier rows, and marks the episode erroneous.
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(packet())
    assert app._start_recording_core()
    assert app.outcome == 'success' and app.record_metadata['outcome'] == 'success'
    app._capture_action_sample(100.0)
    app._report_recording_error('arm_command 源数据已过期')
    assert not app.is_recording
    metadata = json.loads((Path(app.session_dir['record']) / 'metadata.json').read_text())
    assert metadata['status'] == 'error' and metadata['action_count'] == 1
    assert metadata['outcome'] == 'success'
    print('PASS: recording quality errors preserve valid rows and mark error without invoking motion controls')

    app.action_sample_buffer.clear(); app.action_sample_buffer.append(packet())
    assert app._start_recording_core()
    app._capture_action_sample(100.0)
    failed_goal = copy.deepcopy(not_reached)
    failed_goal['data']['gripper_target'].update(status='error', result_error='hardware exception', error='hardware exception')
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(failed_goal)
    try:
        app._capture_action_sample(100.0)
    except RuntimeError as exc:
        app._report_recording_error(str(exc))
    else:
        raise AssertionError('A true gripper error did not stop the recording')
    metadata = json.loads((Path(app.session_dir['record']) / 'metadata.json').read_text())
    assert metadata['status'] == 'error' and metadata['action_count'] == 1 and metadata['outcome'] == 'success'
    actual_failure = metadata['capture_failure_diagnostics']['latest_gripper_target']
    assert actual_failure['status'] == 'error' and actual_failure['result_error'] == 'hardware exception'
    assert actual_failure['result_status'] == 6 and actual_failure['result_success'] is False
    print('PASS: true gripper failures stop the episode and metadata retains the actual raw result, independently of the default task outcome')

    # A partial HDF5 write must never publish a complete-looking canonical JSON.
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(packet())
    assert app._start_recording_core()
    app._capture_action_sample(100.0)
    real_h5_file = mod.h5py.File
    def partial_h5_failure(path, mode):
        partial = real_h5_file(path, mode)
        class PartialFile:
            attrs = partial.attrs
            def __enter__(self): return self
            def create_dataset(self, *args, **kwargs): raise OSError('simulated HDF5 dataset write failure')
            def __exit__(self, *args): partial.close()
        return PartialFile()
    with patch.object(mod.h5py, 'File', partial_h5_failure):
        app.stop_recording()
    failed_record = Path(app.session_dir['record'])
    assert not list((failed_record / 'action').iterdir())
    failed_metadata = json.loads((failed_record / 'metadata.json').read_text())
    assert failed_metadata['status'] == 'error'
    diagnostic = json.loads(next((failed_record / 'diagnostics').glob('*.json')).read_text())
    assert diagnostic['save_status'] == 'error' and diagnostic['metadata']['status'] == 'error'
    assert len(diagnostic['data']) == 1 and diagnostic['task'] == app.task_text
    print('PASS: partial HDF5 failure leaves no canonical pair/temporary files and retains explicitly erroneous raw diagnostics')

    # Failure while publishing the second file must roll back the first publication too.
    app.action_sample_buffer.clear(); app.action_sample_buffer.append(packet())
    assert app._start_recording_core()
    app._capture_action_sample(100.0)
    real_replace = os.replace
    def fail_second_publish(source, destination):
        if str(destination).endswith('.json'): raise OSError('simulated JSON publication failure')
        return real_replace(source, destination)
    with patch.object(mod.os, 'replace', fail_second_publish):
        app.stop_recording()
    failed_record = Path(app.session_dir['record'])
    assert not list((failed_record / 'action').iterdir())
    assert json.loads((failed_record / 'metadata.json').read_text())['status'] == 'error'
    print('PASS: second-file publication failure removes the already-published half pair')

    # The old replay adapter must refuse this new action meaning before spawning anything.
    app.robot_ready, app.action_socket, app.replay_in_progress = True, object(), False
    app._get_replay_index = lambda: 1
    warnings = []
    def forbidden(*args, **kwargs): raise AssertionError('Replay attempted a process or motion operation')
    with patch.object(mod.messagebox, 'showwarning', lambda title, message: warnings.append(message)), \
         patch.object(mod.threading, 'Thread', forbidden), patch.object(app, '_pause_remote_control', forbidden):
        app.start_replay()
    assert warnings and '不发送' in warnings[0]
    app._append_action_history({'timestamp': 100.0, 'gripper_command': 1.0,
                               'gripper_target': {'valid': True, 'target_width_m': 0.07, 'max_width_m': 0.08}})
    assert app.action_history[-1]['gripper'] == 0.07
    print('PASS: schema-2 replay is refused before motion and chart fallback prefers the accepted physical-width target')
