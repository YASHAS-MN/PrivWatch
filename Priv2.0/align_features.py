import os
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Setup
model = YOLO('yolov8n-pose.pt')
base_raw_dir = r"C:\Prototypes\Priv2.0\data\raw_videos"
base_kp_dir = r"C:\Prototypes\Priv2.0\data\keypoints"
target_classes = ['normal', 'fight'] # These are the ones with 68 features

print("🚀 Starting Alignment: Converting legacy videos to 51-feature YOLO skeletons...")

for cls in target_classes:
    raw_path = os.path.join(base_raw_dir, cls)
    kp_path = os.path.join(base_kp_dir, cls)
    os.makedirs(kp_path, exist_ok=True)
    
    videos = [f for f in os.listdir(raw_path) if f.endswith(('.mp4', '.avi'))]
    print(f"🎬 Processing {len(videos)} videos in class: {cls}")
    
    for idx, v_file in enumerate(videos):
        cap = cv2.VideoCapture(os.path.join(raw_path, v_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 16: continue
            
        # Sample 16 frames evenly
        indices = [int(i * (total_frames - 1) / 16) for i in range(16)]
        sequence = []
        
        for f_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break
            
            # Use same eye as live feed
            results = model(frame, imgsz=320, verbose=False)[0]
            if len(results.keypoints.data) > 0:
                kp = results.keypoints.data[0].cpu().numpy().flatten()
                sequence.append(kp[:51]) # Ensure exactly 51
            else:
                sequence.append(np.zeros(51))
        cap.release()
        
        if len(sequence) == 16:
            # This overwrites the old 68-feature files
            np.save(os.path.join(kp_path, f"{cls}_{idx:03d}.npy"), np.array(sequence))

print("✅ Alignment Complete. All datasets are now 51-feature YOLOv8 format.")