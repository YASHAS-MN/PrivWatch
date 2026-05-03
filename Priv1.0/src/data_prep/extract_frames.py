import cv2
import os
import numpy as np

# Configuration - Tuned for 4GB VRAM limits
SEQUENCE_LENGTH = 16  # Number of frames per video
IMAGE_SIZE = (128, 128) # Resize to save memory

def extract_frames(video_path, output_dir, video_id):
    """Extracts exactly SEQUENCE_LENGTH frames from a video, evenly spaced."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < SEQUENCE_LENGTH:
        print(f"Skipping {video_path}: Too few frames ({total_frames})")
        return False
        
    # Calculate indices for evenly spaced frames
    skip_interval = max(int(total_frames / SEQUENCE_LENGTH), 1)
    frame_indices = [i * skip_interval for i in range(SEQUENCE_LENGTH)]
    
    frame_count = 0
    extracted_count = 0
    
    # Create a subfolder for this specific video's frames
    video_out_dir = os.path.join(output_dir, f"video_{video_id}")
    os.makedirs(video_out_dir, exist_ok=True)
    
    while cap.isOpened() and extracted_count < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in frame_indices:
            # Resize frame to our target 128x128 to save memory
            frame_resized = cv2.resize(frame, IMAGE_SIZE)
            
            # Save the frame
            frame_filename = os.path.join(video_out_dir, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame_resized)
            extracted_count += 1
            
        frame_count += 1
        
    cap.release()
    return True

def process_dataset(raw_dir, output_dir):
    """Iterates through raw videos and extracts frames."""
    print(f"Processing videos from {raw_dir}...")
    
    videos = [f for f in os.listdir(raw_dir) if f.endswith(('.mp4', '.avi'))]
    
    if not videos:
        print(f"WARNING: No .mp4 or .avi files found in {raw_dir}!")
        return
        
    for i, video_file in enumerate(videos):
        video_path = os.path.join(raw_dir, video_file)
        success = extract_frames(video_path, output_dir, i)
        
        if success and i % 50 == 0: # Print progress every 50 videos
            print(f"Processed {i}/{len(videos)} videos")

if __name__ == "__main__":
    # Define our paths based on the structure we created
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    RAW_VIOLENCE = os.path.join(BASE_DIR, "data", "raw_videos", "violence")
    RAW_NORMAL = os.path.join(BASE_DIR, "data", "raw_videos", "normal")
    
    FRAMES_VIOLENCE = os.path.join(BASE_DIR, "data", "frames", "violence")
    FRAMES_NORMAL = os.path.join(BASE_DIR, "data", "frames", "normal")
    
    # Execute extraction
    print("--- Starting Violence Class Extraction ---")
    process_dataset(RAW_VIOLENCE, FRAMES_VIOLENCE)
    
    print("\n--- Starting Normal Class Extraction ---")
    process_dataset(RAW_NORMAL, FRAMES_NORMAL)
    
    print("\nFrame extraction complete!")