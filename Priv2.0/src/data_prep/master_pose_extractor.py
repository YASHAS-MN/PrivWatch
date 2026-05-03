import os
import cv2
import numpy as np
import math
from ultralytics import YOLO

# Suppress YOLO's default print statements for a clean terminal
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

print("🚀 Initializing PrivWatch 2.0 Master Pose Extractor...")
model = YOLO('yolov8n-pose.pt') 

SEQUENCE_LENGTH = 16  
FEATURES_PER_FRAME = 68 

def extract_top2_skeletons(results):
    frame_data = np.zeros(FEATURES_PER_FRAME)
    
    if not results or not results[0].keypoints:
        return frame_data 
        
    boxes = results[0].boxes.xywh.cpu().numpy()
    
    # YOLOv8 sometimes returns empty keypoints even if a box is found
    if results[0].keypoints.xyn is None:
        return frame_data
        
    keypoints = results[0].keypoints.xyn.cpu().numpy() 
    
    if len(boxes) == 0 or len(keypoints) == 0:
        return frame_data
        
    areas = [w * h for x, y, w, h in boxes]
    sorted_indices = np.argsort(areas)[::-1] 
    
    # Actor 1
    actor1_idx = sorted_indices[0]
    frame_data[0:34] = keypoints[actor1_idx].flatten() 
    
    # Actor 2
    if len(sorted_indices) > 1:
        actor2_idx = sorted_indices[1]
        frame_data[34:68] = keypoints[actor2_idx].flatten()
        
    return frame_data

def process_class_directory(class_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_dir = os.path.join(base_dir, "data", "raw_videos", class_name)
    kp_dir = os.path.join(base_dir, "data", "keypoints", class_name)
    
    if not os.path.exists(raw_dir):
        print(f"⚠️ Directory not found: {raw_dir}")
        return

    videos = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not videos:
        print(f"📂 {class_name.upper()} is currently empty. Skipping.")
        return
        
    print(f"\n--- ⚙️ Extracting Skeletons for {class_name.upper()} ({len(videos)} videos) ---")
    
    success_count = 0
    for i, video_file in enumerate(videos):
        video_path = os.path.join(raw_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(kp_dir, f"{base_name}.npy")
        
        # Skip if already processed
        if os.path.exists(output_path):
            success_count += 1
            continue
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < SEQUENCE_LENGTH:
            cap.release()
            continue

        skip_interval = max(math.floor(total_frames / SEQUENCE_LENGTH), 1)
        target_frames = [j * skip_interval for j in range(SEQUENCE_LENGTH)]
        
        sequence_data = []
        current_frame = 0
        
        while cap.isOpened() and len(sequence_data) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret: break
                
            if current_frame in target_frames:
                results = model(frame, verbose=False)
                frame_features = extract_top2_skeletons(results)
                sequence_data.append(frame_features)
                
            current_frame += 1
            
        cap.release()
        
        if len(sequence_data) == SEQUENCE_LENGTH:
            np.save(output_path, np.array(sequence_data))
            success_count += 1
            
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(videos)} videos...")

    print(f"✅ {class_name.upper()} COMPLETE: {success_count}/{len(videos)} sequences saved.")

if __name__ == "__main__":
    classes = ["normal", "fight", "collapse", "intrusion"]
    for cls in classes:
        process_class_directory(cls)
    print("\n🏁 ALL EXTRACTIONS FINISHED.")