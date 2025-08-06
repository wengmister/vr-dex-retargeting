import numpy as np
import socket
import threading
import time

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


class VRHandDetector:
    def __init__(self, hand_type="Right", udp_port=9000):
        self.hand_type = hand_type
        self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        self.udp_port = udp_port
        self.latest_landmarks = None
        self.socket = None
        self.listening_thread = None
        self.is_running = False
        
        # Start UDP listener
        self.start_udp_listener()
        
    def detect(self):
        """
        Replace MediaPipe's detect() method
        Must return same format: (num_hands, joint_pos, keypoint_2d, wrist_rotation)
        """
        # Get VR hand tracking data
        vr_landmarks = self.get_vr_hand_landmarks()  # Your VR implementation
        
        if vr_landmarks is None:
            return 0, None, None, None
            
        # Ensure correct format: (21, 3) array
        assert vr_landmarks.shape == (21, 3), f"Expected (21, 3), got {vr_landmarks.shape}"
        
        # Apply same transformations as MediaPipe version
        keypoint_3d_array = vr_landmarks.copy()
        
        # Make wrist the origin (same as MediaPipe processing)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        
        # Estimate hand orientation (reuse MediaPipe's algorithm)
        wrist_rot = VRHandDetector.estimate_frame_from_hand_points(keypoint_3d_array)
        
        # Transform to MANO coordinate system
        joint_pos = keypoint_3d_array @ wrist_rot @ self.operator2mano
        
        return 1, joint_pos, None, wrist_rot  # keypoint_2d=None for VR
    
    def get_vr_hand_landmarks(self) -> np.ndarray:
        """
        Get VR hand landmarks from UDP stream data
        
        Must return: np.ndarray of shape (21, 3) with landmarks in the exact
        same order as MediaPipe's hand model
        
        VR data order: 1x wrist, 4x thumb, 4x index, 4x middle, 4x ring, 4x pinky (21 total)
        MediaPipe order: wrist, thumb(4), index(4), middle(4), ring(4), pinky(4)
        """
        if self.latest_landmarks is None:
            return None
            
        # The VR data ordering matches MediaPipe exactly:
        # Index 0: WRIST
        # Index 1-4: THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP  
        # Index 5-8: INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP
        # Index 9-12: MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP
        # Index 13-16: RING_MCP, RING_PIP, RING_DIP, RING_TIP
        # Index 17-20: PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP
        
        landmarks = self.latest_landmarks.copy()
        
        # Scale down to match MediaPipe range (VR was ~5x larger than expected)
        landmarks *= 1.05
        
        # Coordinate system analysis from comparison:
        # MediaPipe ranges: X[-0.000, 0.050], Y[-0.057, 0.091], Z[0.000, 0.175]
        # VR ranges: X[-0.295, 0.030], Y[-0.233, 0.455], Z[0.000, 0.907]
        # 
        # VR seems to have different axis orientation than MediaPipe
        # Let's try remapping Unity coordinate system to match MediaPipe better
        
        # Create new coordinate mapping
        new_landmarks = landmarks.copy()
        
        # Based on the data patterns, try remapping axes:
        # VR X (left-right) -> MediaPipe X 
        # VR Y (up-down) -> MediaPipe Y
        # VR Z (forward-back) -> MediaPipe Z
        # But with different scaling and orientation
        
        if self.hand_type == "Right":
            # For right hand, flip X to correct mirroring
            new_landmarks[:, 0] = -landmarks[:, 0]
        
        return new_landmarks
    
    def start_udp_listener(self):
        """Start UDP socket listener in a separate thread"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('', self.udp_port))
            self.socket.settimeout(1.0)
            self.is_running = True
            
            self.listening_thread = threading.Thread(target=self._udp_listener, daemon=True)
            self.listening_thread.start()
        except Exception as e:
            print(f"Failed to start UDP listener: {e}")
    
    def _udp_listener(self):
        """Background thread to listen for UDP data"""
        while self.is_running:
            try:
                data, _ = self.socket.recvfrom(4096)
                decoded_data = data.decode('utf-8').strip()
                
                if decoded_data.startswith("Right landmarks:"):
                    landmarks = self._parse_landmark_data(decoded_data)
                    if landmarks is not None:
                        self.latest_landmarks = landmarks
                else:
                    # Debug: show what we're receiving that doesn't match
                    if len(decoded_data) > 0:
                        # ignore right wrist data:
                        if "Right wrist" in decoded_data:
                            continue
                        print(f"Received non-landmark data: '{decoded_data[:50]}...'")
                    else:
                        print("Received empty data")
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"UDP listener error: {e}")
                break
    
    def _parse_landmark_data(self, data_string):
        """Parse the UDP landmark data string into numpy array"""
        try:
            # Remove "Right landmarks:" prefix and split by commas
            values_str = data_string.replace("Right landmarks:", "").strip()
            raw_values = values_str.split(",")
            
            # Filter out empty strings and convert to float
            values = []
            for x in raw_values:
                x_clean = x.strip()
                if x_clean:  # Only process non-empty strings
                    values.append(float(x_clean))
            
            # Should have 21 landmarks * 3 coordinates = 63 values
            if len(values) != 63:
                print(f"Expected 63 values, got {len(values)} from data: {data_string[:100]}...")
                return None
            
            # Reshape into (21, 3) array
            landmarks = np.array(values).reshape(21, 3)
            return landmarks
            
        except Exception as e:
            print(f"Failed to parse landmark data: {e}")
            print(f"Raw data: {data_string[:200]}...")
            return None
    
    def stop_udp_listener(self):
        """Stop the UDP listener and cleanup"""
        self.is_running = False
        if self.socket:
            self.socket.close()
        if self.listening_thread:
            self.listening_thread.join(timeout=2.0)

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def draw_skeleton_on_image(self, image, keypoint_2d, style="default"):
        """
        Draw skeleton on image. For VR, keypoint_2d is None, so return image unchanged.
        """
        if keypoint_2d is None:
            return image
        return image
