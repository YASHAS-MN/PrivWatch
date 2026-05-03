import cv2
import numpy as np
import math
from ultralytics import YOLO

# Load the YOLOv8 Pose model (nano version is blazing fast for edge devices)
print("🚀 Loading YOLOv8-Pose Nano...")
model = YOLO('yolov8n-pose.pt') 

# The fixed sequence length your LSTM will expect
SEQUENCE_LENGTH = 16  
# 17 joints * 2 (x,y) = 34. Two actors = 68 coordinates per frame.
FEATURES_PER_FRAME = 68 

def extract_top2_skeletons(results):
    """
    Parses YOLO output to find the 2 most prominent people in the frame.
    Returns a flat array of 68 coordinates.
    """
    frame_data = np.zeros(FEATURES_PER_FRAME)
    
    if not results or not results[0].keypoints:
        return frame_data # Return all zeros if no one is in the frame
        
    # Get the bounding boxes to sort by size (largest = closest to camera)
    boxes = results[0].boxes.xywh.cpu().numpy()
    keypoints = results[0].keypoints.xyn.cpu().numpy() # Normalized (0 to 1) coordinates
    
    if len(boxes) == 0:
        return frame_data
        
    # Sort detected humans by the area of their bounding box (width * height)
    areas = [w * h for x, y, w, h in boxes]
    sorted_indices = np.argsort(areas)[::-1] # Descending order
    
    # Extract Actor 1
    actor1_idx = sorted_indices[0]
    # Flatten the 17 (x,y) pairs into 34 numbers
    actor1_kpts = keypoints[actor1_idx].flatten() 
    frame_data[0:34] = actor1_kpts
    
    # Extract Actor 2 (if they exist)
    if len(sorted_indices) > 1:
        actor2_idx = sorted_indices[1]
        actor2_kpts = keypoints[actor2_idx].flatten()
        frame_data[34:68] = actor2_kpts
        
    return frame_data

def process_video_with_temporal_skip(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📹 Video Loaded: {total_frames} frames @ {fps} FPS")
    
    if total_frames < SEQUENCE_LENGTH:
        print("⚠️ Video too short.")
        return None

    # YOUR TEMPORAL SKIPPING MATH
    # E.g., if total_frames = 160, skip_interval = 10. We take every 10th frame.
    skip_interval = math.floor(total_frames / SEQUENCE_LENGTH)
    target_frames = [i * skip_interval for i in range(SEQUENCE_LENGTH)]
    
    print(f"⏱️ Skipping logic applied. Extracting at frame indices: {target_frames}")
    
    sequence_data = []
    current_frame = 0
    
    while cap.isOpened() and len(sequence_data) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame in target_frames:
            # Run YOLOv8-pose on this specific frame
            results = model(frame, verbose=False)
            
            # Extract Top 2 Actors
            frame_features = extract_top2_skeletons(results)
            sequence_data.append(frame_features)
            
        current_frame += 1
        
    cap.release()
    
    if len(sequence_data) == SEQUENCE_LENGTH:
        tensor = np.array(sequence_data)
        print(f"✅ Success! Extracted Matrix Shape: {tensor.shape} (16 frames, 68 physics features)")
        return tensor
    else:
        print("❌ Failed to extract a full sequence.")
        return None

if __name__ == "__main__":
    # Test it out! Put a test video in your root folder and run this.
    test_video = "test_video.avi" 
    import os
    if os.path.exists(test_video):
        process_video_with_temporal_skip(test_video)
    else:
        print(f"Drop a '{test_video}' file in this directory and run me to test the engine.")