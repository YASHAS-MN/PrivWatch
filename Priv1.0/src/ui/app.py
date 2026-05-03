import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque
import os

# --- 1. MODEL ARCHITECTURE (Must match training exactly) ---
class PoseLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_layers=2, num_classes=4):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes) # 4 Classes

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# --- 2. SYSTEM INITIALIZATION ---
st.set_page_config(page_title="PrivWatch AI", layout="wide")
st.title("🛡️ PrivWatch: 4-Class Privacy-Preserving Surveillance")
st.markdown("Analyzing purely anonymized kinematic data. No facial recognition. No identity tracking.")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseLSTM().to(device)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    weight_path = os.path.join(base_dir, "models", "saved_weights", "pose_lstm_4class.pth")
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# MediaPipe Tracker
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# The 4-Class Taxonomy
CLASSES = {0: "NORMAL", 1: "VIOLENCE DETECTED", 2: "SUDDEN COLLAPSE", 3: "INTRUSION / TRESPASSING"}
# BGR Colors for OpenCV: Green, Red, Orange, Purple
COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 165, 255), 3: (255, 0, 255)} 

# --- 3. UI DASHBOARD ---
col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    st.subheader("Live System Status")
    status_text = st.empty()
    st.markdown("---")
    st.write("**Active Modules:**")
    st.write("🟢 33-Point Skeletal Tracker")
    st.write("🟢 Temporal Synchronization")
    st.write("🟢 4-Class LSTM Brain")

start_button = st.button("Start Live Surveillance")
stop_button = st.button("Stop Feed")

# --- 4. REAL-TIME INFERENCE LOOP ---
if start_button:
    cap = cv2.VideoCapture(0) # 0 for default laptop webcam
    sequence_buffer = deque(maxlen=16)
    frame_counter = 0 
    
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
            
        frame = cv2.flip(frame, 1) # Mirror image
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Temporal Fix: Only process every 3rd frame to match the 3-second training window
        if frame_counter % 3 == 0:
            if results.pose_landmarks:
                frame_features = []
                for lm in results.pose_landmarks.landmark:
                    frame_features.extend([lm.x, lm.y, lm.z, lm.visibility])
                sequence_buffer.append(frame_features)
            else:
                # Pad with zeros if no person is seen to keep time moving
                sequence_buffer.append(list(np.zeros(132)))
        
        # Draw Skeleton regardless of frame skipping (looks smoother for the user)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        # Inference Trigger
        if len(sequence_buffer) == 16:
            input_tensor = torch.FloatTensor([sequence_buffer]).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                class_idx = predicted_class.item()
                conf_score = confidence.item() * 100
                
                label = CLASSES[class_idx]
                color = COLORS[class_idx]
                
                # Dynamic UI Alert Box
                cv2.rectangle(frame, (0, 0), (640, 60), color, -1)
                cv2.putText(frame, f"{label} ({conf_score:.1f}%)", (15, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                
                # Streamlit Sidebar Updates
                if class_idx == 0: status_text.success(f"✅ {label}")
                elif class_idx == 1: status_text.error(f"🚨 {label}!")
                elif class_idx == 2: status_text.warning(f"⚠️ {label}!")
                elif class_idx == 3: status_text.error(f"🛑 {label}!")

        frame_counter += 1
        
        # Push to Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()