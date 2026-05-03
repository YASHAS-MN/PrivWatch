import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import deque
import mediapipe as mp
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="PrivWatch Dashboard", layout="wide")
st.title("🛡️ PrivWatch: Privacy-Preserving Urban Surveillance")
st.markdown("Upload a video and select a processing pipeline to detect anomalies in real-time.")

# --- HARDWARE & MODELS SETUP ---
@st.cache_resource # This prevents Streamlit from reloading models on every click
def load_hardware_and_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. RAW Model Class
    class RawLSTM(nn.Module):
        def __init__(self, input_size=2048, hidden_size=64, num_layers=1):
            super(RawLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.sigmoid(self.fc(lstm_out[:, -1, :]))

    # 2. POSE Model Class
    class PoseLSTM(nn.Module):
        def __init__(self, input_size=132, hidden_size=64, num_layers=1):
            super(PoseLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.sigmoid(self.fc(lstm_out[:, -1, :]))

    # Load ResNet
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet_extractor.eval()
    
    # Load Weights
    BASE_DIR = os.path.dirname(__file__)
    raw_model = RawLSTM().to(device)
    raw_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "saved_weights", "raw_lstm_model.pth"), map_location=device))
    raw_model.eval()
    
    pose_model = PoseLSTM().to(device)
    pose_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "models", "saved_weights", "pose_lstm_model.pth"), map_location=device))
    pose_model.eval()
    
    return device, resnet_extractor, raw_model, pose_model

device, resnet_extractor, raw_model, pose_model = load_hardware_and_models()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("System Controls")
pipeline_choice = st.sidebar.radio("Select Processing Pipeline:", ("RAW (Conventional)", "PRIVACY (Kinematic)"))
uploaded_file = st.sidebar.file_uploader("Upload Surveillance Video (MP4/AVI)", type=['mp4', 'avi'])

# --- MAIN ENGINE LOGIC ---
if uploaded_file is not None:
    # Save uploaded file temporarily so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    if st.sidebar.button("Start Processing Stream"):
        st.markdown(f"### Live Feed: {pipeline_choice} Pipeline")
        
        # Streamlit placeholders to update video/text dynamically
        status_text = st.empty()
        video_placeholder = st.empty()
        
        cap = cv2.VideoCapture(video_path)
        SEQUENCE_LENGTH = 16
        frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        frame_skip = 2
        frame_count = 0
        
        # Pipeline specific setups
        if pipeline_choice == "RAW (Conventional)":
            preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            pose_extractor = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                display_frame = frame.copy()
                
                if frame_count % frame_skip == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if pipeline_choice == "RAW (Conventional)":
                        pil_img = Image.fromarray(rgb_frame)
                        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                        features = resnet_extractor(input_tensor).squeeze().cpu().numpy()
                        frame_buffer.append(features)
                        
                    elif pipeline_choice == "PRIVACY (Kinematic)":
                        # Create black frame
                        display_frame = np.zeros(frame.shape, dtype=np.uint8)
                        results = pose_extractor.process(rgb_frame)
                        frame_landmarks = []
                        
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
                            for lm in results.pose_landmarks.landmark:
                                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                        else:
                            frame_landmarks = list(np.zeros(132))
                        
                        frame_buffer.append(frame_landmarks)
                        # Convert black frame to RGB for Streamlit
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Inference
                    if len(frame_buffer) == SEQUENCE_LENGTH:
                        sequence_tensor = torch.FloatTensor(np.array(frame_buffer)).unsqueeze(0).to(device)
                        
                        if pipeline_choice == "RAW (Conventional)":
                            prediction = raw_model(sequence_tensor).item()
                        else:
                            prediction = pose_model(sequence_tensor).item()
                            
                        # Update UI
                        if prediction > 0.5:
                            status_text.markdown(f"### 🚨 **STATUS: VIOLENCE DETECTED ({prediction*100:.1f}%)**")
                        else:
                            status_text.markdown(f"### ✅ **STATUS: NORMAL ({100 - prediction*100:.1f}%)**")
                
                # If RAW, we need to convert BGR to RGB for Streamlit to show colors correctly
                if pipeline_choice == "RAW (Conventional)":
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Push the frame to the webpage
                video_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                
        cap.release()
        st.success("Video Processing Complete.")