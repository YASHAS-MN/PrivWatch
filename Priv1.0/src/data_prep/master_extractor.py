import cv2
import os
import numpy as np
import mediapipe as mp

# --- TIER 2: KINEMATIC EXTRACTION ENGINE ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
SEQUENCE_LENGTH = 16

def extract_kinematics(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < SEQUENCE_LENGTH:
        cap.release()
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

def process_all_classes():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # ALL 4 CLASSES LOCKED IN
    classes = ["normal", "fight", "collapse", "intrusion"]
    
    for cls in classes:
        print(f"\n--- Extracting {cls.upper()} Skeletons ---")
        raw_dir = os.path.join(base_dir, "data", "raw_videos", cls)
        kp_dir = os.path.join(base_dir, "data", "keypoints", cls)
        
        os.makedirs(kp_dir, exist_ok=True)
        
        if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
            print(f"⚠️ No raw videos found for {cls.upper()}. Skipping...")
            continue
            
        videos = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        valid_count = 0
        
        for i, video_file in enumerate(videos):
            video_path = os.path.join(raw_dir, video_file)
            base_name = os.path.splitext(video_file)[0]
            output_path = os.path.join(kp_dir, f"{base_name}.npy")
            
            # Skips already processed files (Normal, Fight, Collapse)
            if os.path.exists(output_path):
                valid_count += 1
                continue
                
            if extract_kinematics(video_path, output_path):
                valid_count += 1
                
            if i % 10 == 0 and i > 0:
                print(f"   Processed {i}/{len(videos)} {cls} videos...")
                
        print(f"✅ Extracted {valid_count} pure kinematic sequences for {cls.upper()}.")

if __name__ == "__main__":
    print("🚀 INITIATING TIER 2: KINEMATIC EXTRACTION...")
    process_all_classes()
    print("\n🏁 EXTRACTION COMPLETE.")