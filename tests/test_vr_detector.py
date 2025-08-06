#!/usr/bin/env python3

import sys
import os
import time

# Add the example/vector_retargeting directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'example', 'vector_retargeting'))

from vr_hand_detector import VRHandDetector

def test_vr_detector():
    """Test VR hand detector by listening for real UDP data on port 9000"""
    print("VR Hand Detector Test - Listening on port 9000")
    print("=" * 50)
    print("Make sure your VR system is sending data to port 9000")
    print("Press Ctrl+C to stop\n")
    
    # Initialize detector
    detector = VRHandDetector(hand_type="Right", udp_port=9000)
    
    # Wait for initialization
    time.sleep(1.0)
    
    try:
        print("Waiting for VR data...")
        last_data_time = time.time()
        
        while True:
            result = detector.detect()
            num_hands, joint_pos, _, _ = result
            
            current_time = time.time()
            
            if num_hands > 0 and joint_pos is not None:
                print(f"\n[{time.strftime('%H:%M:%S')}] VR Data Received:")
                print(f"  ✓ Hands detected: {num_hands}")
                print(f"  ✓ Joint positions shape: {joint_pos.shape}")
                print(f"  ✓ Wrist position: [{joint_pos[0][0]:.4f}, {joint_pos[0][1]:.4f}, {joint_pos[0][2]:.4f}]")
                print(f"  ✓ Thumb tip: [{joint_pos[4][0]:.4f}, {joint_pos[4][1]:.4f}, {joint_pos[4][2]:.4f}]")
                print(f"  ✓ Index tip: [{joint_pos[8][0]:.4f}, {joint_pos[8][1]:.4f}, {joint_pos[8][2]:.4f}]")
                
                # Show raw landmarks too
                raw_landmarks = detector.get_vr_hand_landmarks()
                if raw_landmarks is not None:
                    print(f"  ✓ Raw wrist: [{raw_landmarks[0][0]:.4f}, {raw_landmarks[0][1]:.4f}, {raw_landmarks[0][2]:.4f}]")
                
                last_data_time = current_time
            else:
                # Show waiting message every 5 seconds if no data
                if current_time - last_data_time > 5.0:
                    print(f"[{time.strftime('%H:%M:%S')}] Still waiting for VR data...")
                    last_data_time = current_time
            
            time.sleep(0.1)  # 10Hz polling
            
    except KeyboardInterrupt:
        print("\n\nStopping VR detector...")
    finally:
        detector.stop_udp_listener()
        print("VR detector stopped.")

if __name__ == "__main__":
    test_vr_detector()