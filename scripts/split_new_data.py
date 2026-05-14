import os
import random
import shutil
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "Normal_New"
SPLIT_BASE_DIR = PROJECT_ROOT / "data" / "split"

# Split Ratios: 80% Train, 10% Val, 10% Test
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

def distribute_data():
    if not SOURCE_DIR.exists():
        print(f"❌ Error: Could not find source folder: {SOURCE_DIR}")
        return

    # Get all video files
    videos = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    random.shuffle(videos)
    
    total_videos = len(videos)
    print(f"📦 Found {total_videos} new videos. Shuffling and splitting...")

    # Calculate split indices
    train_end = int(total_videos * SPLIT_RATIOS["train"])
    val_end = train_end + int(total_videos * SPLIT_RATIOS["val"])

    splits = {
        "train": videos[:train_end],
        "val": videos[train_end:val_end],
        "test": videos[val_end:]
    }

    # Move files to their new homes
    for split_name, file_list in splits.items():
        # Target directory: e.g., data/split/train/Normal
        target_dir = SPLIT_BASE_DIR / split_name / "Normal"
        target_dir.mkdir(parents=True, exist_ok=True)

        for video_file in file_list:
            src_path = SOURCE_DIR / video_file
            dst_path = target_dir / video_file
            shutil.copy2(src_path, dst_path) # using copy2 to preserve original files just in case
            
        print(f"✅ Copied {len(file_list)} videos to {split_name}/Normal")

    print("🏁 Data split complete! You are ready to preprocess.")

if __name__ == "__main__":
    distribute_data()