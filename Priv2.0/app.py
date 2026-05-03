import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
import statistics
import tempfile

# --- 1. LOAD THE BRAIN (LSTM) ---
class ActionLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=64, num_layers=2, num_classes=4):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@st.cache_resource
def load_models():
    # Load YOLO (The Eye)
    yolo_model = YOLO('yolov8n-pose.pt')
    
    # Load LSTM (The Brain)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = ActionLSTM().to(device)
    lstm_model.load_state_dict(torch.load('models/action_model.pth', map_location=device))
    lstm_model.eval() # Set to evaluation mode
    
    return yolo_model, lstm_model, device

yolo_model, lstm_model, device = load_models()
CLASSES = ['Normal Activity', 'Fight Detected', 'Collapse Detected', 'Intrusion Detected']
COLORS = [(0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 0)] # Green, Red, Orange, Blue

# --- 2. STREAMLIT UI ---
st.set_page_config(page_title="PrivWatch AI", layout="wide")
st.title("🛡️ PrivWatch: Privacy-Preserving Surveillance")
st.markdown("**Real-time Anomaly Detection via Skeletal Geometry**")

uploaded_file = st.file_uploader("Upload a Surveillance Video (MP4/AVI)", type=['mp4', 'avi'])

if uploaded_file is not None:
    # Save uploaded video to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    status_text = st.empty()
    
    # The Sliding Window for Sequence
    sequence_buffer = deque(maxlen=16)
    prediction_buffer = deque(maxlen=10) # Temporal smoothing
    
    # 🛠️ THE RECOVERY FIX: Temporal Tracking
    frame_counter = 0 
    temporal_stride = 5 # Process every 5th frame (matches training temporal spread)
    
    st.markdown("### 🔴 Live Feed Analysis")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_counter += 1
        
        # Draw standard UI text (keeps the video running smoothly)
        frame = cv2.resize(frame, (320, 240))
        
        # 🛠️ Only extract pose and predict every 5th frame
        if frame_counter % temporal_stride == 0:
            results = yolo_model(frame, imgsz=320, verbose=False)[0]
            
            # 🛠️ CONFIDENCE GATE: Check if a clear human is actually present
            if len(results.keypoints.data) > 0:
                conf_scores = results.keypoints.conf[0].cpu().numpy()
                mean_conf = np.mean(conf_scores)
                
                if mean_conf > 0.4: # Only process if YOLO is confident
                    kp = results.keypoints.data[0].cpu().numpy().flatten()
                    pose_data = np.zeros(51)
                    if len(kp) >= 51:
                        pose_data = kp[:51]
                    else:
                        pose_data[:len(kp)] = kp
                    
                    # Draw Skeleton
                    for i in range(0, len(pose_data), 3):
                        x, y, conf = pose_data[i], pose_data[i+1], pose_data[i+2]
                        if conf > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

                    sequence_buffer.append(pose_data)
                    
                    if len(sequence_buffer) == 16:
                        seq_tensor = torch.tensor(np.array(sequence_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = lstm_model(seq_tensor)
                            _, predicted = torch.max(outputs.data, 1)
                            prediction_buffer.append(predicted.item())
                else:
                    sequence_buffer.clear()
            else:
                sequence_buffer.clear()

        # Determine what text to show based on the smoothed buffer
        if len(prediction_buffer) > 0 and len(sequence_buffer) == 16:
            smooth_idx = statistics.mode(prediction_buffer)
            current_action = CLASSES[smooth_idx]
            color = COLORS[smooth_idx]
        elif len(sequence_buffer) > 0:
            current_action = f"Tracking... {len(sequence_buffer)}/16"
            color = (255, 255, 255)
        else:
            current_action = "No Target Detected"
            color = (150, 150, 150)
        
        cv2.putText(frame, f"STATUS: {current_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
        
    cap.release()