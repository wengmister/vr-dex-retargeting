import cv2
import numpy as np
import socket
import threading
import time
from vr_hand_detector import VRHandDetector
from single_hand_detector import SingleHandDetector

class HandDataComparison:
    def __init__(self, hand_type="Right", udp_port=9000):
        self.hand_type = hand_type
        self.udp_port = udp_port
        
        # Initialize VR detector
        self.vr_detector = VRHandDetector(hand_type=hand_type, udp_port=udp_port)
        
        # Initialize MediaPipe detector
        self.mp_detector = SingleHandDetector(hand_type=hand_type, selfie=False)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        self.joint_names = [
            "WRIST",
            "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP", 
            "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
            "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]
    
    def run_comparison(self):
        print("Starting VR vs MediaPipe hand data comparison...")
        print("Press 'q' to quit, 'c' to compare current frame data")
        print("Make sure to click on the OpenCV window to focus it for keyboard input!")
        print("=" * 80)
        
        frame_count = 0
        while True:
            frame_count += 1
            # Capture webcam frame
            success, frame = self.cap.read()
            if not success:
                continue
                
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get MediaPipe detection
            _, mp_joint_pos, keypoint_2d, _ = self.mp_detector.detect(rgb)
            
            # Get VR detection
            _, vr_joint_pos, _, _ = self.vr_detector.detect()
            
            # Draw MediaPipe skeleton if available
            if keypoint_2d is not None:
                frame = self.mp_detector.draw_skeleton_on_image(frame, keypoint_2d, style="default")
            
            # Add status text
            status_text = f"Frame: {frame_count}"
            if mp_joint_pos is not None:
                status_text += " | MP: Hand detected"
            else:
                status_text += " | MP: No hand"
                
            if vr_joint_pos is not None:
                status_text += " | VR: Hand detected"
            else:
                status_text += " | VR: No hand"
                
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to compare, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("MediaPipe Hand Detection", frame)
            
            # Make sure the window is focused for key input
            cv2.setWindowProperty("MediaPipe Hand Detection", cv2.WND_PROP_TOPMOST, 1)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("\n[COMPARISON TRIGGERED]")
                self.compare_joint_data(mp_joint_pos, vr_joint_pos)
                print("[Press 'c' again for another comparison, 'q' to quit]\n")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.vr_detector.stop_udp_listener()
    
    def compare_joint_data(self, mp_joints, vr_joints):
        print("\n" + "=" * 80)
        print("JOINT DATA COMPARISON")
        print("=" * 80)
        
        if mp_joints is None and vr_joints is None:
            print("No hand detected by either system")
            return
        elif mp_joints is None:
            print("MediaPipe: No hand detected")
            if vr_joints is not None:
                print(f"VR: Hand detected with shape {vr_joints.shape}")
                self.print_vr_data(vr_joints)
            return
        elif vr_joints is None:
            print("VR: No hand detected")
            if mp_joints is not None:
                print(f"MediaPipe: Hand detected with shape {mp_joints.shape}")
                self.print_mp_data(mp_joints)
            return
        
        # Both systems have data
        print(f"MediaPipe joints shape: {mp_joints.shape}")
        print(f"VR joints shape: {vr_joints.shape}")
        print()
        
        # Compare joint by joint
        print(f"{'Joint':<15} {'MP_X':<8} {'MP_Y':<8} {'MP_Z':<8} {'VR_X':<8} {'VR_Y':<8} {'VR_Z':<8} {'Diff':<10}")
        print("-" * 80)
        
        for i in range(21):
            mp_pos = mp_joints[i] if i < len(mp_joints) else [0, 0, 0]
            vr_pos = vr_joints[i] if i < len(vr_joints) else [0, 0, 0]
            
            diff = np.linalg.norm(mp_pos - vr_pos) if len(mp_pos) == 3 and len(vr_pos) == 3 else 0
            
            print(f"{self.joint_names[i]:<15} "
                  f"{mp_pos[0]:<8.3f} {mp_pos[1]:<8.3f} {mp_pos[2]:<8.3f} "
                  f"{vr_pos[0]:<8.3f} {vr_pos[1]:<8.3f} {vr_pos[2]:<8.3f} "
                  f"{diff:<10.3f}")
        
        # Overall statistics
        print("\nOVERALL STATISTICS:")
        print(f"MediaPipe coordinate ranges:")
        print(f"  X: [{mp_joints[:, 0].min():.3f}, {mp_joints[:, 0].max():.3f}]")
        print(f"  Y: [{mp_joints[:, 1].min():.3f}, {mp_joints[:, 1].max():.3f}]")
        print(f"  Z: [{mp_joints[:, 2].min():.3f}, {mp_joints[:, 2].max():.3f}]")
        
        print(f"VR coordinate ranges:")
        print(f"  X: [{vr_joints[:, 0].min():.3f}, {vr_joints[:, 0].max():.3f}]")
        print(f"  Y: [{vr_joints[:, 1].min():.3f}, {vr_joints[:, 1].max():.3f}]")
        print(f"  Z: [{vr_joints[:, 2].min():.3f}, {vr_joints[:, 2].max():.3f}]")
        
        print("=" * 80)
    
    def print_mp_data(self, joints):
        print("\nMediaPipe joint data:")
        for i, joint_name in enumerate(self.joint_names):
            if i < len(joints):
                pos = joints[i]
                print(f"  {joint_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    def print_vr_data(self, joints):
        print("\nVR joint data:")
        for i, joint_name in enumerate(self.joint_names):
            if i < len(joints):
                pos = joints[i]
                print(f"  {joint_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

if __name__ == "__main__":
    comparison = HandDataComparison(hand_type="Right")
    comparison.run_comparison()