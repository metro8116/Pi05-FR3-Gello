# Component Provenance and Licensing

English · [简体中文](LICENSE_zh.md)

This repository integrates components from multiple sources. **It does not declare a single MIT or Apache-2.0 license for the entire directory.** Existing copyright headers, LICENSE files, and NOTICE files are retained and take precedence for their respective files and components.

| Component | Source | License / status |
| --- | --- | --- |
| `ros2_ws/src/franka_gello_state_publisher` | Local GELLO ROS project | Component MIT LICENSE retained |
| `ros2_ws/src/franka_fr3_arm_controllers` | Franka ROS examples and local controller modifications | Apache-2.0 LICENSE/NOTICE retained |
| `ros2_ws/src/franka_gripper_manager` | Local Franka ROS project | Apache-2.0 LICENSE/NOTICE retained |
| `ros2_ws/src/pi05_gripper_driver` | Independent driver derived from Franka gripper | Apache-2.0 LICENSE/NOTICE retained |
| `capture/` | [renshaojie233/data-collection](https://github.com/renshaojie233/data-collection) and current experimental modifications | No explicit LICENSE was found at the upstream repository root during migration; see `capture/SOURCE.txt` |
| `src/fr3_pi05` and training entry points | The user's existing Pi05-train project | Migrated local functionality and integration changes; the rights holder must specify license terms before publication |
| `deploy/` | The user's existing Pi05-deploy project | Migrated local functionality and integration changes; derived drivers retain their component licenses |
| Converter and integration scripts/tutorials | The user's workflow and this integration | Third-party material without a license grant has not been relicensed; confirm licensing for original contributions before publication |

GELLO upstream: [wuphilipp/gello_software](https://github.com/wuphilipp/gello_software). Its root MIT license text is retained in `licenses/gello_software-MIT.txt` as a provenance record; it is not extended to material from other sources.

Runtime dependencies are not bundled with the source. LeRobot, Transformers, PyTorch, libfranka, franka_ros2, the RealSense SDK, and FFmpeg/imageio-ffmpeg each remain subject to their own licenses. Building libfranka from source and distributing images must also comply with the license and notice requirements for the corresponding version. The `lerobot/pi05_base` model, PaliGemma tokenizer, data, and actual videos each have their own use and distribution conditions; the code repository's licensing cannot replace those conditions.

This work created a local code repository only; it has not been uploaded to GitHub. Before public release, confirm redistribution terms with the collection upstream's rights holder, or replace that collection component with an authorized or independently reimplemented version. Also specify the license for the project's original files. This notice records existing facts and does not mean that third-party permission has already been obtained.
