import argparse
import sys
import time
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from privwatch.inference import run_inference


def run_privacy_inference(cap):
    return run_inference(cap)


def main():
    parser = argparse.ArgumentParser(description="Run PrivWatch privacy inference on a video.")
    parser.add_argument("video_path", help="Path to an input video file")
    args = parser.parse_args()

    print("Privacy model running...")
    start_time = time.time()
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    result = run_privacy_inference(cap)
    end_time = time.time()

    print(f"\nFINAL RESULT: {result}")
    print(f"\nLatency: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
