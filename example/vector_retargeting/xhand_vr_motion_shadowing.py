import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from vr_hand_detector import VRHandDetector
from xhand_interface import XHandInterface 
import sys


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):

    # First of all, open device 
    device_identifier = {}

    xhand_robot = XHandInterface(hand_id=0, position=0.1, mode=3)

    while True:
        device_identifier['protocol'] = 'RS485'
        # You can use enumerate_devices('RS485') to read serial port list information, choose ttyUSB prefixed port
        # Get serial port list, choose ttyUSB*
        xhand_robot.enumerate_devices('RS485')
        device_identifier["serial_port"] = '/dev/ttyUSB0'
        device_identifier['baud_rate'] = 3000000
        if xhand_robot.open_device(device_identifier):
            break
        else:
            sys.exit(1)

    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    
    # Initialize VR detector with robot name for robot-specific adaptations
    detector = VRHandDetector(hand_type=hand_type, robot_name=robot_name, use_tcp=True)

    loader.load_multiple_collisions_from_file = True

    # Only xhand in config now
    loader.scale = 1.1

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "xhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "bidexhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)
    
    # Debug logging: print joint name mappings
    logger.info("=== JOINT MAPPING DEBUG ===")
    logger.info(f"Robot name: {robot_name}")
    logger.info(f"Retargeting type: {retargeting.optimizer.retargeting_type}")
    logger.info(f"Number of Sapien joints: {len(sapien_joint_names)}")
    logger.info(f"Number of retargeting joints: {len(retargeting_joint_names)}")
    logger.info("Sapien joint names (order as loaded from URDF):")
    for i, name in enumerate(sapien_joint_names):
        logger.info(f"  [{i}]: {name}")
    logger.info("Retargeting joint names (order from config):")
    for i, name in enumerate(retargeting_joint_names):
        logger.info(f"  [{i}]: {name}")
    logger.info("Mapping array (retargeting -> sapien indices):")
    logger.info(f"  {retargeting_to_sapien}")
    logger.info("=== END JOINT MAPPING DEBUG ===")
    
    # Also get the target joint names from the optimizer for comparison
    if hasattr(retargeting.optimizer, 'target_joint_names'):
        target_joint_names = retargeting.optimizer.target_joint_names
        logger.info(f"Target joint names from optimizer: {target_joint_names}")
        logger.info(f"Number of target joints: {len(target_joint_names)}")
    
    # XHand interface expects 12 joints
    expected_xhand_joints = 12
    logger.info(f"XHand interface expects {expected_xhand_joints} joints")
    if len(joint_positions if 'joint_positions' in locals() else sapien_joint_names) != expected_xhand_joints:
        logger.warning(f"Joint count mismatch! Sapien has {len(sapien_joint_names)} joints, XHand expects {expected_xhand_joints}")
        
    # Create mapping from retargeting output to desired XHand joint order
    # Desired XHand order: [thumb_bend, thumb_rota1, thumb_rota2, index_bend, index_joint1, index_joint2, 
    #                       mid_joint1, mid_joint2, ring_joint1, ring_joint2, pinky_joint1, pinky_joint2]
    desired_xhand_joint_names = [
        'right_hand_thumb_bend_joint', 'right_hand_thumb_rota_joint1', 'right_hand_thumb_rota_joint2',
        'right_hand_index_bend_joint', 'right_hand_index_joint1', 'right_hand_index_joint2',
        'right_hand_mid_joint1', 'right_hand_mid_joint2',
        'right_hand_ring_joint1', 'right_hand_ring_joint2',
        'right_hand_pinky_joint1', 'right_hand_pinky_joint2'
    ]
    
    # Create mapping from retargeting output indices to desired XHand indices
    retargeting_to_xhand = []
    for desired_joint in desired_xhand_joint_names:
        if desired_joint in retargeting_joint_names:
            retargeting_to_xhand.append(retargeting_joint_names.index(desired_joint))
        else:
            logger.error(f"Desired joint {desired_joint} not found in retargeting joints!")
            retargeting_to_xhand.append(0)  # fallback
    
    retargeting_to_xhand = np.array(retargeting_to_xhand)
    
    logger.info("=== XHAND JOINT MAPPING ===")
    logger.info("Desired XHand joint order -> Retargeting index:")
    for i, (desired_joint, retarg_idx) in enumerate(zip(desired_xhand_joint_names, retargeting_to_xhand)):
        logger.info(f"  XHand[{i}] {desired_joint} <- Retargeting[{retarg_idx}]")
    logger.info("=== END SETUP DEBUG INFO ===")

    while True:
        try:
            bgr = queue.get(timeout=5)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            return

        _, joint_pos, keypoint_2d, _ = detector.detect()

        if joint_pos is None:
            # logger.debug(f"{hand_type} hand is not detected.")
            pass
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            # Log retargeting input
            logger.debug(f"Retargeting input ref_value shape: {ref_value.shape}")
            logger.debug(f"Retargeting input ref_value: {ref_value.flatten()}")
            
            qpos = retargeting.retarget(ref_value)
            
            # Log raw retargeting output
            logger.debug(f"Raw retargeting output qpos shape: {qpos.shape}")
            logger.debug(f"Raw retargeting output qpos: {qpos}")
            
            # Log joint names with their corresponding values
            logger.debug("=== RETARGETING OUTPUT WITH JOINT NAMES ===")
            for i, (joint_name, joint_value) in enumerate(zip(retargeting_joint_names, qpos)):
                logger.debug(f"  [{i}] {joint_name}: {joint_value:.6f}")
            
            robot.set_qpos(qpos[retargeting_to_sapien])

            # Log remapped joint positions for Sapien
            joint_positions = qpos[retargeting_to_sapien]
            logger.debug("=== SAPIEN JOINT POSITIONS (after remapping) ===")
            for i, (joint_name, joint_value) in enumerate(zip(sapien_joint_names, joint_positions)):
                logger.debug(f"  Sapien[{i}] {joint_name}: {joint_value:.6f}")
            
            print("Joint positions:", joint_positions)

            # Map retargeting output to correct XHand joint order
            xhand_joint_positions = qpos[retargeting_to_xhand]

            xhand_joint_positions[3] = -xhand_joint_positions[3]  # Invert index bend for XHand
            
            # Log XHand command mapping
            logger.debug("=== XHAND COMMAND MAPPING (Corrected Order) ===")
            logger.debug("XHand Index -> Joint Name -> Value:")
            for i, (joint_name, joint_value) in enumerate(zip(desired_xhand_joint_names, xhand_joint_positions)):
                logger.debug(f"  XHand[{i}] {joint_name}: {joint_value:.6f}")
                xhand_robot._hand_command.finger_command[i].position = joint_value
            
            # If we have fewer joints than expected, log which ones are missing
            if len(xhand_joint_positions) < 12:
                logger.debug(f"Missing {12 - len(xhand_joint_positions)} joints, keeping default positions for fingers [{len(xhand_joint_positions)}:11]")
                
            xhand_robot.send_command()


        for _ in range(2):
            viewer.render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


def main(
    camera_path: Optional[str] = None,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.
    Uses xhand robot with dexpilot retargeting mode and right hand only.

    Args:
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    robot_name = RobotName.xhand
    retargeting_type = RetargetingType.dexpilot
    hand_type = HandType.right
    
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
