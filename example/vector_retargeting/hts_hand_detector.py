"""
Hand detector that uses hand-tracking-sdk's HTSClient to receive
hand tracking data from Meta Quest's Hand Tracking Streamer app.

Provides the same interface as VRHandDetector so it can be used as
a drop-in replacement in the retargeting pipeline.
"""

import threading
import numpy as np

from hand_tracking_sdk import (
    HTSClient,
    HTSClientConfig,
    HandFilter,
    HandFrame,
    HandSide,
    StreamEvent,
    StreamOutput,
    TransportMode,
)

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


def _hand_frame_to_landmarks(frame: HandFrame) -> np.ndarray:
    """Convert a HandFrame's landmarks to a (21, 3) numpy array.

    HTS landmarks are wrist-relative in Unity left-handed coordinates.
    """
    return np.array(frame.landmarks.points, dtype=np.float64)


class HTSHandDetector:
    """Hand detector backed by hand-tracking-sdk's HTSClient.

    Replaces VRHandDetector's raw UDP socket parsing with the proper
    hand-tracking-sdk client, giving us reliable frame assembly,
    transport flexibility (UDP/TCP), and coordinate utilities.
    """

    def __init__(
        self,
        hand_type: str = "Right",
        robot_name: str | None = None,
        transport: TransportMode = TransportMode.UDP,
        host: str = "0.0.0.0",
        port: int = 9000,
    ):
        self.hand_type = hand_type
        self.robot_name = robot_name
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )

        # Map hand_type to hand-tracking-sdk's HandFilter
        if hand_type == "Right":
            hand_filter = HandFilter.RIGHT
        elif hand_type == "Left":
            hand_filter = HandFilter.LEFT
        else:
            hand_filter = HandFilter.BOTH

        self._latest_frame: HandFrame | None = None
        self._lock = threading.Lock()

        # Create HTSClient
        self._client = HTSClient(
            HTSClientConfig(
                transport_mode=transport,
                host=host,
                port=port,
                output=StreamOutput.FRAMES,
                hand_filter=hand_filter,
            )
        )

        # Start background thread to consume frames
        self._running = True
        self._thread = threading.Thread(target=self._consume_frames, daemon=True)
        self._thread.start()

    def _consume_frames(self):
        """Background thread that pulls frames from HTSClient."""
        try:
            for event in self._client.iter_events():
                if not self._running:
                    break
                if isinstance(event, HandFrame):
                    with self._lock:
                        self._latest_frame = event
        except Exception as e:
            if self._running:
                print(f"HTSClient error: {e}")

    def detect(self):
        """Get the latest hand detection result.

        Returns the same format as VRHandDetector:
            (num_hands, joint_pos, keypoint_2d, wrist_rotation)

        joint_pos is (21, 3) in MANO convention, wrist-relative.
        """
        with self._lock:
            frame = self._latest_frame

        if frame is None:
            return 0, None, None, None

        # Extract (21, 3) landmarks from frame
        landmarks = _hand_frame_to_landmarks(frame)

        # Convert coordinate system: Unity left-handed to right-handed
        # In Unity left-hand coords, Y is up and Z is forward.
        # For right-hand, we flip the Y axis (same as what the SDK's
        # convert_landmarks_unity_left_to_right does internally).
        if self.hand_type == "Right":
            landmarks[:, 0] = -landmarks[:, 0]

        # Scale to match expected coordinate range
        landmarks *= 1.05

        # Make wrist the origin
        landmarks = landmarks - landmarks[0:1, :]

        # Estimate hand orientation frame
        wrist_rot = self.estimate_frame_from_hand_points(landmarks)

        # Transform to MANO coordinate system
        joint_pos = landmarks @ wrist_rot @ self.operator2mano

        # Apply robot-specific adaptations
        if self.robot_name and "sharpa_ha4" in self.robot_name:
            # HA4 has similar proportions to human hand, slight scale up
            joint_pos *= 1.2

        return 1, joint_pos, None, wrist_rot

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """Compute 3D coordinate frame from hand keypoints.

        Uses wrist, index MCP, and middle MCP to estimate palm orientation.
        Same algorithm as VRHandDetector / MediaPipe convention.
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)
        normal = v[2, :]

        # Gram-Schmidt orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # Ensure z points from pinky to index side
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def stop(self):
        """Stop the background frame consumer."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def draw_skeleton_on_image(self, image, keypoint_2d, style="default"):
        """No-op for compatibility — VR tracking has no 2D image."""
        return image
