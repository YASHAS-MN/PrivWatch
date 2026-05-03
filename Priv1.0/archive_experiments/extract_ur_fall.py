import cv2
import os
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
SEQUENCE_LENGTH = 16

def extract_direct_kinematics(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQUENCE_LENGTH:
        return False
        
    skip_interval = max(int(total_frames / SEQUENCE_LENGTH), 1)
    frame_indices = [i * skip_interval for i in range(SEQUENCE_LENGTH)]
    
    frame_count, extracted_count = 0, 0
    video_keypoints = []
    
    while cap.isOpened() and extracted_count < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret: break
            
        if frame_count in frame_indices:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                frame_landmarks = []
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                video_keypoints.append(frame_landmarks)
            else:
                video_keypoints.append(list(np.zeros(132)))
            extracted_count += 1
        frame_count += 1
    cap.release()
    
    if len(video_keypoints) == SEQUENCE_LENGTH:
        np.save(output_path, np.array(video_keypoints))
        return True
    return False

def process_dataset(raw_dir, kp_dir, class_name):
    print(f"\n--- Extracting {class_name} Data ---")
    os.makedirs(kp_dir, exist_ok=True)
    videos = [f for f in os.listdir(raw_dir) if f.endswith(('.mp4', '.avi'))]
    
    valid_count = 0
    for i, video_file in enumerate(videos):
        video_path = os.path.join(raw_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(kp_dir, f"{base_name}.npy")
        
        if extract_direct_kinematics(video_path, output_path):
            valid_count += 1
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(videos)} videos")
            
    print(f"Successfully extracted {valid_count} {class_name} sequences.")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # Normal Paths
    RAW_NORMAL = os.path.join(BASE_DIR, "data", "raw_videos", "normal")
    KP_NORMAL = os.path.join(BASE_DIR, "data", "keypoints", "normal")
    
    # Collapse Paths
    RAW_COLLAPSE = os.path.join(BASE_DIR, "data", "raw_videos", "collapse")
    KP_COLLAPSE = os.path.join(BASE_DIR, "data", "keypoints", "collapse")
    
    process_dataset(RAW_NORMAL, KP_NORMAL, "Normal")
    process_dataset(RAW_COLLAPSE, KP_COLLAPSE, "Collapse")