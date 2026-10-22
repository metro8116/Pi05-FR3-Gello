import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def generate_nodes(context):
    config_file_name = LaunchConfiguration("config_file").perform(context)
    package_config_dir = FindPackageShare("franka_gripper_manager").perform(context)
    config_file = os.path.join(package_config_dir, "config", config_file_name)
    configs = load_yaml(config_file)
    nodes = []
    for item_name, config in configs.items():
        parameters = {"namespace": str(config["namespace"])}
        for param_name in (
            "control_mode",
            "grasp_action_topic",
            "grasp_force_n",
            "grasp_force_limit_n",
            "command_speed_limit_m_s",
            "force_configuration_verified",
            "grasp_epsilon_inner_m",
            "grasp_epsilon_outer_m",
            "close_trigger_width_m",
            "release_trigger_width_m",
            "measurement_max_age_s",
            "move_action_topic",
            "homing_action_topic",
            "gripper_command_topic",
            "joint_states_topic",
            "gripper_move_speed",
            "command_deadband_percent",
            "max_width_fallback",
            "invert_gripper_command",
            "home_on_startup",
        ):
            if param_name in config:
                parameters[param_name] = config[param_name]
        nodes.append(
            Node(
                package="franka_gripper_manager",
                executable="franka_gripper_client",
                name="franka_gripper_client",
                namespace=str(config["namespace"]),
                output="screen",
                parameters=[parameters],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value="example_fr3_config_franka_hand.yaml",
                description="Path to the franka_gripper_manager configuration file to load",
            ),
            OpaqueFunction(function=generate_nodes),
        ]
    )
