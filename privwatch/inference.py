import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from privwatch.model import ActionModel
from privwatch.paths import RAW_MODEL_PATH

# ================= CONFIG =================
SEQ_LEN = 16
IMG_SIZE = 112
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIGHT_THRESHOLD = 0.35
COLLAPSE_THRESHOLD = 0.35

CLASS_MAP = {
    0: "Normal",
    1: "Fight",
    2: "Collapse",
}


def load_model():
    model = ActionModel().to(DEVICE)
    checkpoint = torch.load(RAW_MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("=== MODEL FORENSIC INFO ===")
        print(f"epoch: {checkpoint.get('epoch')}")
        print(f"validation accuracy: {checkpoint.get('val_accuracy')}")
        print(f"timestamp: {checkpoint.get('timestamp')}")
        print(f"dataset summary: {checkpoint.get('dataset_summary')}")
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(cap):
    model = load_model()
    
    buffer = []
    fight_probs = []
    normal_probs = []
    collapse_probs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            continue

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        buffer.append(frame)

        if len(buffer) == SEQ_LEN:
            clip = torch.from_numpy(np.array(buffer)).permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(clip.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(clip.device)
            clip = (clip - mean) / std
            clip = clip.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(clip)
                probs = torch.softmax(outputs, dim=1)[0]

                p_fight = probs[1].item()
                p_normal = probs[0].item()
                p_collapse = probs[2].item()

                fight_probs.append(p_fight)
                normal_probs.append(p_normal)
                collapse_probs.append(p_collapse)

            buffer.pop(0)  # sliding window

    cap.release()

    if len(fight_probs) > 0:
        final_fight = sum(fight_probs) / len(fight_probs)
        final_normal = sum(normal_probs) / len(normal_probs)
        final_collapse = sum(collapse_probs) / len(collapse_probs)

        if final_fight >= FIGHT_THRESHOLD:
            final_output = "ALERT - Fight detected"
        elif final_collapse >= COLLAPSE_THRESHOLD:
            final_output = "ALERT - Collapse detected"
        else:
            final_output = "Normal activity"

        print("\n===== FINAL VIDEO CONFIDENCE =====")
        print(f"Fight: {final_fight:.2f}")
        print(f"Normal: {final_normal:.2f}")
        print(f"Collapse: {final_collapse:.2f}")
        print("===============================")

        print("\n===== FINAL DECISION =====")
        print(f"Fight threshold: {FIGHT_THRESHOLD}")
        print(f"Collapse threshold: {COLLAPSE_THRESHOLD}")
        print(f"Selected output: {final_output}")
        print("====================")

        return final_output

    return "No frames processed"


def main():
    parser = argparse.ArgumentParser(description="Run PrivWatch raw inference on a video.")
    parser.add_argument("video_path", help="Path to an input video file")
    args = parser.parse_args()

    print("Model loaded. Running inference...")
    start_time = time.time()
    
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    
    result = run_inference(cap)
    end_time = time.time()

    print(f"\nFINAL RESULT: {result}")
    print(f"\nLatency: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
