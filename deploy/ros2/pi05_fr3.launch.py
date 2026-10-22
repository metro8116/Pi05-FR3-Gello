"""PI05 FR3 driver session. No GELLO node and no old gripper client.

The impedance controller is loaded INACTIVE. Only the local bridge can arm it.
All critical process exits stop this entire launch session.
"""
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.actions import RegisterEventHandler, EmitEvent
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from config import load_config


def _shutdown_if_failed(event, context):
    if event.returncode:
        return [EmitEvent(event=Shutdown(reason='PI05 controller spawner failed'))]
    return []


def _nodes(context):
    config_path = Path(LaunchConfiguration('config_file').perform(context)).resolve()
    cfg = load_config(config_path)
    robot = cfg['robot']
    urdf = Path(get_package_share_directory('franka_description')) / 'robots/fr3/fr3.urdf.xacro'
    description = xacro.process_file(str(urdf), mappings={
        'ros2_control': 'true', 'arm_id': robot['arm_id'], 'arm_prefix': '',
        'robot_ip': robot['ip'], 'hand': 'true', 'use_fake_hardware': 'false',
        'fake_sensor_commands': 'false',
    }).toprettyxml(indent='  ')
    controllers = str(ROOT / 'configs/controllers_pi05.yaml')
    driver = Node(
        package='controller_manager', executable='ros2_control_node', output='screen',
        parameters=[controllers, {'robot_description': description, 'arm_id': robot['arm_id'],
                                 'namespace': '', 'load_gripper': 'true'}],
        remappings=[('joint_states', 'franka/joint_states')])
    robot_state = Node(package='robot_state_publisher', executable='robot_state_publisher',
                       parameters=[{'robot_description': description}], output='screen')
    broadcasts = [
        Node(package='controller_manager', executable='spawner',
             arguments=[name, '--controller-manager-timeout', '30'], output='screen')
        # PI05 consumes JointState only; the unused full Franka state publisher
        # can fail its realtime publisher lock without helping this control path.
        for name in ('joint_state_broadcaster',)
    ]
    impedance = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=[cfg['bridge']['controller_name'], '--inactive', '--controller-manager-timeout', '30'])
    gripper_config = Path(get_package_share_directory('franka_gripper')) / 'config/franka_gripper_node.yaml'
    # The deployment overlay isolates blocking Stop callbacks from fresh state
    # publication; the system/collection gripper package remains untouched.
    gripper = Node(package='pi05_gripper_driver', executable='pi05_gripper_node', name='franka_gripper',
                   output='screen', parameters=[str(gripper_config), {
                       'robot_ip': robot['ip'],
                       'joint_names': [robot['arm_id'] + '_finger_joint' + str(i) for i in (1, 2)],
                   }])
    bridge = ExecuteProcess(cmd=['/usr/bin/python3', str(ROOT / 'fr3_policy_bridge.py'),
                                 '--config', str(config_path)], output='screen')
    actions = [robot_state, driver, gripper, *broadcasts, impedance, bridge]
    # Registered first, so even startup failures trigger shutdown.
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=process,
        on_exit=[EmitEvent(event=Shutdown(reason='Critical PI05 driver/bridge exited'))]))
        for process in (driver, bridge, gripper)]
    handlers += [RegisterEventHandler(OnProcessExit(target_action=process, on_exit=_shutdown_if_failed))
                 for process in (*broadcasts, impedance)]
    return handlers + actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=str(ROOT / 'configs/deploy.yaml')),
        OpaqueFunction(function=_nodes),
    ])
