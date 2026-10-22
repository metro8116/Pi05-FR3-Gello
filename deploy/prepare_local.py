"""Check/build the policy controller overlay, without starting any ROS node."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from config import REPO_ROOT

HOST_WS = REPO_ROOT/'ros2_ws'
CONTAINER_WS = '/workspace/ros2'
OVERLAY = CONTAINER_WS + '/pi05_deploy_overlay'
GRIPPER_SOURCE = HOST_WS/'src/pi05_gripper_driver'


def docker_base():
    return ['docker','-H',os.environ.get('PI05_DOCKER_HOST','unix:///var/run/docker.sock')]


def container_name():
    return os.environ.get('PI05_CONTAINER','pi05-fr3-gello-ros2-1')


def source_digest():
    root = HOST_WS/'src/franka_fr3_arm_controllers'
    digest = hashlib.sha256()
    names = ['src/joint_impedance_controller.cpp','include/franka_fr3_arm_controllers/joint_impedance_controller.hpp','CMakeLists.txt']
    names += [str(p.relative_to(root)) for p in sorted((root/'test').rglob('*')) if p.suffix in ('.cpp','.hpp')]
    for name in names:
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    for path in sorted(GRIPPER_SOURCE.rglob('*')):
        if path.is_file():
            digest.update(('pi05_gripper_driver/'+str(path.relative_to(GRIPPER_SOURCE))).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def check_overlay():
    stamp = HOST_WS/'pi05_deploy_overlay/controller_build.json'
    if not stamp.is_file():
        raise RuntimeError('Policy controller overlay is not prepared. Run deployment --prepare first (no robot activation).')
    record = json.loads(stamp.read_text())
    if (record.get('source_sha256') != source_digest() or not record.get('tests_passed')
            or not record.get('gripper_driver_tests_passed')):
        raise RuntimeError('Controller source changed or tests did not pass. Re-run --prepare.')
    if not (HOST_WS/'pi05_deploy_overlay/install/setup.bash').is_file():
        raise RuntimeError('Controller overlay setup.bash is missing; re-run --prepare')
    if not (HOST_WS/'pi05_deploy_overlay/install/pi05_gripper_driver/lib/pi05_gripper_driver/pi05_gripper_node').is_file():
        raise RuntimeError('Isolated gripper driver is missing; re-run --prepare')
    return record


def prepare(build=False):
    result=subprocess.run(docker_base()+['inspect','--format','{{.State.Running}}',container_name()],
                          check=True, capture_output=True, text=True)
    if result.stdout.strip() != 'true':
        raise RuntimeError('The existing ROS2 container is not running; open/start that dev container first')
    if not build:
        return check_overlay()
    overlay_host=HOST_WS/'pi05_deploy_overlay'
    if not overlay_host.exists():
        overlay_host.mkdir()
    elif not os.access(overlay_host,os.W_OK):
        # Docker may have created this NEW overlay directory as root. Change
        # only its top directory so the host can publish its build stamp.
        subprocess.run(docker_base()+['exec',container_name(),'chown',
                                      f'{os.getuid()}:{os.getgid()}',OVERLAY],check=True)
    stamp=HOST_WS/'pi05_deploy_overlay/controller_build.json'
    if stamp.exists():stamp.unlink()
    # Copy only our uniquely named package; never replace the installed system
    # franka_gripper or collection workspace sources/install.
    gripper_copy = overlay_host/'src/pi05_gripper_driver'
    if gripper_copy.exists():shutil.rmtree(gripper_copy)
    gripper_copy.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(GRIPPER_SOURCE,gripper_copy)
    # Builds a separate overlay: the existing collection install remains intact.
    source = ('source /opt/ros/humble/setup.bash && '
              'source /opt/ros/humble/franka/setup.bash && '
              'source /workspace/ros2/install/setup.bash && cd /workspace/ros2 && ')
    command = source + (
        'colcon --log-base pi05_deploy_overlay/log build '
        '--base-paths src/franka_fr3_arm_controllers pi05_deploy_overlay/src/pi05_gripper_driver '
        '--build-base pi05_deploy_overlay/build --install-base pi05_deploy_overlay/install '
        '--cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON && '
        'source pi05_deploy_overlay/install/setup.bash && '
        "ROS_DOMAIN_ID=192 ctest --test-dir pi05_deploy_overlay/build/franka_fr3_arm_controllers "
        "-R '^test_joint_impedance_controller$' --no-tests=error --output-on-failure && "
        "ROS_DOMAIN_ID=192 ROS_LOCALHOST_ONLY=1 ctest --test-dir pi05_deploy_overlay/build/pi05_gripper_driver "
        "-R '^test_stop_isolation$' --no-tests=error --output-on-failure")
    subprocess.run(docker_base()+['exec',container_name(),'bash','-lc',command], check=True)
    record = {'source_sha256':source_digest(),'tests_passed':True,'gripper_driver_tests_passed':True,'overlay':OVERLAY,
              'gripper_package':'pi05_gripper_driver',
              'test_scope':'C++ controller behavior and injected-backend gripper Stop/state isolation tests',
              'package_lint_note':'Full package clang_format has pre-existing differences; no unrelated formatting edits.'}
    stamp=HOST_WS/'pi05_deploy_overlay/controller_build.json'
    stamp.write_text(json.dumps(record,indent=2)+'\n')
    return record


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build',action='store_true')
    args=ap.parse_args(argv)
    try:
        print(json.dumps(prepare(args.build),indent=2))
        return 0
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
