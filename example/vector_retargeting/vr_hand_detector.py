import numpy as np
import socket
import threading
import time
import subprocess

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


def adaptive_retargeting_xhand(landmarks):
    """
    Apply adaptive pinky retargeting specifically for XHand robot.
    Compensates for human-to-robot finger length differences by using
    adaptive scaling based on finger extension state.
    
    Args:
        landmarks: np.ndarray of shape (21, 3) with hand landmarks
        
    Returns:
        np.ndarray of shape (21, 3) with retargeted landmarks
    """
    landmarks = landmarks.copy()
    
    pinky_mcp = 17   # PINKY_MCP (base)
    pinky_pip = 18   # PINKY_PIP
    pinky_dip = 19   # PINKY_DIP  
    pinky_tip = 20   # PINKY_TIP
    
    # Adaptive scaling based on finger curl state
    # Calculate finger extension (distance from MCP to TIP)
    pinky_extension = np.linalg.norm(landmarks[pinky_tip] - landmarks[pinky_mcp])
    
    # Scale more when extended (for reaching), less when curled (for fist-making)
    # Tuned specifically for XHand robot kinematics
    max_extension = 0.10
    min_extension = 0.03
    
    # Normalize extension ratio (0.0 = fully curled, 1.0 = fully extended)
    extension_ratio = np.clip((pinky_extension - min_extension) / (max_extension - min_extension), 0.0, 1.0)
    
    # Adaptive scaling: more scaling when extended, less when curled
    base_scale = 1.2   # Minimum scaling for curled positions
    max_scale = 2.2    # Maximum scaling for extended positions
    
    adaptive_scale = base_scale + (max_scale - base_scale) * extension_ratio
    
    # Apply same adaptive scaling to all segments
    mcp_to_pip_scale = adaptive_scale
    pip_to_dip_scale = adaptive_scale  
    dip_to_tip_scale = adaptive_scale
    
    # Apply progressive scaling along kinematic chain
    # Start from MCP (base remains unchanged) and extend each segment
    
    # Extend MCP->PIP segment
    mcp_to_pip_vector = landmarks[pinky_pip] - landmarks[pinky_mcp]
    landmarks[pinky_pip] = landmarks[pinky_mcp] + mcp_to_pip_vector * mcp_to_pip_scale
    
    # Extend PIP->DIP segment (using new PIP position)
    pip_to_dip_vector = landmarks[pinky_dip] - landmarks[pinky_pip]  
    landmarks[pinky_dip] = landmarks[pinky_pip] + pip_to_dip_vector * pip_to_dip_scale
    
    # Extend DIP->TIP segment (using new DIP position)
    dip_to_tip_vector = landmarks[pinky_tip] - landmarks[pinky_dip]
    landmarks[pinky_tip] = landmarks[pinky_dip] + dip_to_tip_vector * dip_to_tip_scale
    
    return landmarks


class VRHandDetector:
    def __init__(self, hand_type="Right", udp_port=9000, robot_name=None, use_tcp=False, tcp_port=8000):
        self.hand_type = hand_type
        self.robot_name = robot_name
        self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        self.udp_port = udp_port
        self.use_tcp = use_tcp
        self.tcp_port = tcp_port
        self.latest_landmarks = None
        self.socket = None
        self.tcp_connection = None
        self.tcp_client_address = None
        self.listening_thread = None
        self.is_running = False
        
        # Message counting for periodic reporting
        self.valid_message_count = 0
        self.last_report_time = time.time()
        self.report_interval = 10.0  # Report every 10 seconds
        
        # Start listener (UDP or TCP)
        if self.use_tcp:
            self.start_tcp_listener()
        else:
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
        
        # Apply robot-specific retargeting if specified
        if self.robot_name and "xhand" in self.robot_name:
            joint_pos = adaptive_retargeting_xhand(joint_pos)
        elif self.robot_name and "bidexhand" in self.robot_name:
            # bidexhand has a long wrist offset. Scaling up the whole hand to compensate
            joint_pos *= 1.5
        
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
        
        # Scale to match MediaPipe coordinate range
        landmarks *= 1.05
        
        # Convert coordinate system: Unity left-handed to MediaPipe right-handed
        if self.hand_type == "Right":
            # For right hand, flip X to correct mirroring
            landmarks[:, 0] = -landmarks[:, 0]
        
        return landmarks
    
    def start_tcp_listener(self):
        """Start TCP socket listener in a separate thread"""
        try:
            # Setup adb reverse port forwarding first
            if not self.setup_adb_reverse(self.tcp_port):
                print(f"adb reverse setup failed, but continuing with TCP socket setup")
                print(f"You may need to run 'adb reverse tcp:{self.tcp_port} tcp:{self.tcp_port}' manually")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('localhost', self.tcp_port))
            self.socket.listen(1)  # Allow one connection
            self.is_running = True
            
            self.listening_thread = threading.Thread(target=self._tcp_listener, daemon=True)
            self.listening_thread.start()
            print(f"Successfully bound VR TCP socket to localhost:{self.tcp_port}")
        except Exception as e:
            print(f"Failed to start TCP listener: {e}")
            # Clean up adb reverse if socket setup failed
            self.cleanup_adb_reverse(self.tcp_port)
            raise
    
    def start_udp_listener(self):
        """Start UDP socket listener in a separate thread"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('', self.udp_port))
            self.socket.settimeout(1.0)
            self.is_running = True
            
            self.listening_thread = threading.Thread(target=self._udp_listener, daemon=True)
            self.listening_thread.start()
            print(f"Successfully bound VR UDP socket to port {self.udp_port}")
        except Exception as e:
            print(f"Failed to start UDP listener: {e}")
    
    def _tcp_listener(self):
        """Background thread to listen for TCP data"""
        while self.is_running:
            try:
                # Wait for a connection if we don't have one
                if self.tcp_connection is None:
                    print("Waiting for TCP connection...")
                    self.tcp_connection, self.tcp_client_address = self.socket.accept()
                    self.tcp_connection.settimeout(1.0)  # 1 second timeout for recv
                    print(f"TCP connection established from {self.tcp_client_address}")
                
                # Try to receive data
                try:
                    data = self.tcp_connection.recv(4096)
                    if not data:
                        # Connection closed by client
                        print("TCP client disconnected")
                        self.tcp_connection.close()
                        self.tcp_connection = None
                        self.tcp_client_address = None
                        continue
                    
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
                    # No data received within timeout
                    continue
                    
                except ConnectionResetError:
                    print("TCP connection reset by client")
                    self.tcp_connection.close()
                    self.tcp_connection = None
                    self.tcp_client_address = None
                    continue
                    
            except Exception as e:
                print(f"Error in TCP listener: {e}")
                if self.tcp_connection:
                    self.tcp_connection.close()
                    self.tcp_connection = None
                    self.tcp_client_address = None
                time.sleep(1.0)  # Wait before retrying
    
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
            # Check if message starts with "Right wrist" - suppress these exceptions
            if "Right wrist:" in data_string:
                return None
            
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
            
            # Increment valid message count and check for periodic reporting
            self.valid_message_count += 1
            current_time = time.time()
            if current_time - self.last_report_time >= self.report_interval:
                messages_per_second = self.valid_message_count / (current_time - self.last_report_time)
                print(f"Valid messages per second: {messages_per_second:.1f}")
                self.valid_message_count = 0
                self.last_report_time = current_time
            
            return landmarks
            
        except Exception as e:
            # Suppress exceptions for "Right wrist" messages
            if "Right wrist:" in data_string or "could not convert string to float: '" in str(e) and "Right wrist:" in str(e):
                return None
            print(f"Failed to parse landmark data: {e}")
            print(f"Raw data: {data_string[:200]}...")
            return None
    
    def setup_adb_reverse(self, port):
        """Setup adb reverse port forwarding"""
        try:
            # First check if adb is available
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("adb command not found. Please install Android SDK platform-tools.")
                return False
            
            # Check if device is connected
            if "device" not in result.stdout:
                print("No Android device connected via adb.")
                return False
            
            # Setup reverse port forwarding
            cmd = ['adb', 'reverse', f'tcp:{port}', f'tcp:{port}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"Successfully setup adb reverse tcp:{port} tcp:{port}")
                return True
            else:
                print(f"Failed to setup adb reverse: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("adb command timed out")
            return False
        except FileNotFoundError:
            print("adb command not found. Please install Android SDK platform-tools.")
            return False
        except Exception as e:
            print(f"Error setting up adb reverse: {e}")
            return False

    def cleanup_adb_reverse(self, port):
        """Remove adb reverse port forwarding"""
        try:
            cmd = ['adb', 'reverse', '--remove', f'tcp:{port}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"Successfully removed adb reverse tcp:{port}")
            else:
                print(f"Failed to remove adb reverse: {result.stderr}")
                
        except Exception as e:
            print(f"Error cleaning up adb reverse: {e}")
    
    def stop_listener(self):
        """Stop the listener and cleanup"""
        self.is_running = False
        
        # Close TCP connection if exists
        if self.tcp_connection:
            self.tcp_connection.close()
        
        # Close main socket
        if self.socket:
            self.socket.close()
        
        # Clean up adb reverse if using TCP
        if self.use_tcp:
            self.cleanup_adb_reverse(self.tcp_port)
            
        # Wait for thread to finish
        if self.listening_thread:
            self.listening_thread.join(timeout=2.0)
    
    def stop_udp_listener(self):
        """Stop the UDP listener and cleanup (deprecated, use stop_listener instead)"""
        print("Warning: stop_udp_listener is deprecated, use stop_listener instead")
        self.stop_listener()

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
        _ = style  # Suppress unused parameter warning
        if keypoint_2d is None:
            return image
        return image
    
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        if hasattr(self, 'use_tcp') and self.use_tcp and hasattr(self, 'tcp_port'):
            self.cleanup_adb_reverse(self.tcp_port)
