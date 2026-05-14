import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# CONFIG
SEQ_LEN = 16
IMG_SIZE = 112
CLIPS_PER_VIDEO = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "data" / "split"
DST_ROOT = PROJECT_ROOT / "data" / "clips"

CLASSES = ["Fight", "Normal", "Collapse"]
SPLITS = ["train", "val", "test"]


def extract_clips(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return []

    clips = []

    for _ in range(CLIPS_PER_VIDEO):
        if len(frames) >= SEQ_LEN:
            start = random.randint(0, len(frames) - SEQ_LEN)
            clip = frames[start:start + SEQ_LEN]
        else:
            clip = frames.copy()
            while len(clip) < SEQ_LEN:
                clip.append(clip[-1])

        clip = np.array(clip, dtype=np.uint8)
        clips.append(clip)

    return clips


def process():
    summary = {}

    for split in SPLITS:
        print(f"\n🔹 Processing {split}...\n")
        summary[split] = {}

        for cls in CLASSES:
            src_dir = SRC_ROOT / split / cls
            dst_dir = DST_ROOT / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            videos = [v for v in os.listdir(src_dir) if v.endswith((".mp4", ".avi", ".mov"))]

            processed = 0
            skipped = 0
            clips_saved = 0

            print(f"➡️ {cls} ({len(videos)} videos)")

            for vid in tqdm(videos):
                video_path = src_dir / vid

                clips = extract_clips(video_path)

                if len(clips) == 0:
                    skipped += 1
                    continue

                for i, clip in enumerate(clips):
                    save_path = dst_dir / f"{vid.split('.')[0]}_{i}.pt"
                    torch.save(torch.tensor(clip), save_path)
                    clips_saved += 1

                processed += 1

            summary[split][cls] = {
                "videos_total": len(videos),
                "processed": processed,
                "skipped": skipped,
                "clips": clips_saved
            }

    # 📊 FINAL REPORT
    print("\n\n📊 FINAL DATASET SUMMARY\n")

    for split in summary:
        print(f"\n===== {split.upper()} =====")

        for cls in summary[split]:
            s = summary[split][cls]

            print(
                f"{cls:10} | Total: {s['videos_total']:4} | "
                f"Processed: {s['processed']:4} | "
                f"Skipped: {s['skipped']:4} | "
                f"Clips: {s['clips']:4}"
            )

if __name__ == "__main__":
    process()
