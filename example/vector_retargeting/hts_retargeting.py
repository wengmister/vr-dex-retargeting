"""
Real-time retargeting from Hand Tracking Streamer (Meta Quest) to a robot hand.

Usage:
    # HA4 right hand, UDP on port 9000 (default HTS config):
    python hts_retargeting.py --robot-name sharpa_ha4 --hand-type right

    # HA4 right hand, TCP wired via ADB:
    python hts_retargeting.py --robot-name sharpa_ha4 --hand-type right \
        --transport tcp_server --host 0.0.0.0 --port 8000

    # Shadow hand, left hand:
    python hts_retargeting.py --robot-name shadow --hand-type left --port 9000
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

from hand_tracking_sdk import TransportMode

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from hts_hand_detector import HTSHandDetector


def main(
    robot_name: RobotName = RobotName.sharpa_ha4,
    retargeting_type: RetargetingType = RetargetingType.dexpilot,
    hand_type: HandType = HandType.right,
    transport: str = "udp",
    host: str = "0.0.0.0",
    port: int = 9000,
    urdf_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    print_interval: float = 0.5,
):
    """
    Real-time retargeting from HTS hand tracking to robot joint angles.

    Args:
        robot_name: Target robot hand.
        retargeting_type: Retargeting algorithm (vector, position, dexpilot).
        hand_type: Which hand to track (left or right).
        transport: HTS transport mode: "udp", "tcp_server", or "tcp_client".
        host: Network host to listen on.
        port: Network port.
        urdf_dir: Directory containing robot URDFs. If None, auto-detected.
        config_path: Override config file path. If None, uses default for robot.
        print_interval: How often to print joint angles (seconds).
    """
    # Resolve transport mode
    transport_map = {
        "udp": TransportMode.UDP,
        "tcp_server": TransportMode.TCP_SERVER,
        "tcp_client": TransportMode.TCP_CLIENT,
    }
    transport_mode = transport_map.get(transport)
    if transport_mode is None:
        raise ValueError(f"Unknown transport: {transport}. Use: udp, tcp_server, tcp_client")

    # Resolve config
    if config_path is None:
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    config_path = str(config_path)

    # Resolve URDF directory
    if urdf_dir is None:
        # For HA4, use the URDFs from sharpawave_manus
        if robot_name == RobotName.sharpa_ha4:
            urdf_dir = str(
                Path(__file__).resolve().parents[3]
                / "sharpawave_manus"
                / "retargeting_alg_release_V3.2"
                / "retargeting_alg_release"
                / "urdf"
            )
        else:
            urdf_dir = str(
                Path(__file__).resolve().parent.parent.parent / "assets" / "robots" / "hands"
            )

    print(f"Config:    {config_path}")
    print(f"URDF dir:  {urdf_dir}")
    print(f"Robot:     {robot_name.name}")
    print(f"Hand:      {hand_type.name}")
    print(f"Transport: {transport} on {host}:{port}")
    print()

    # Build retargeting pipeline
    RetargetingConfig.set_default_urdf_dir(urdf_dir)
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    # Map config name to determine hand side string
    hand_type_str = "Right" if hand_type == HandType.right else "Left"

    # Get robot name string for robot-specific adaptations in detector
    robot_name_str = str(config_path).split("/")[-1].replace("_dexpilot.yml", "").replace(".yml", "")

    # Start HTS hand detector
    detector = HTSHandDetector(
        hand_type=hand_type_str,
        robot_name=robot_name_str,
        transport=transport_mode,
        host=host,
        port=port,
    )

    print(f"Listening for {hand_type_str} hand data on {transport}://{host}:{port} ...")
    print("Press Ctrl+C to stop.\n")

    last_print = 0.0
    frame_count = 0

    try:
        while True:
            num_hands, joint_pos, _, _ = detector.detect()

            if joint_pos is None:
                time.sleep(0.005)
                continue

            # Compute retargeting reference value
            retargeting_type_str = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices

            if retargeting_type_str == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            # Run retargeting optimization
            qpos = retargeting.retarget(ref_value)

            frame_count += 1
            now = time.time()

            # Periodic status print
            if now - last_print >= print_interval:
                joint_names = retargeting.joint_names
                angles_deg = np.degrees(qpos)

                print(f"--- Frame {frame_count} ---")
                for name, angle in zip(joint_names, angles_deg):
                    print(f"  {name:30s} {angle:+7.1f} deg")
                print()

                last_print = now

            # Don't spin faster than ~200Hz
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()
        retargeting.verbose()


if __name__ == "__main__":
    tyro.cli(main)
