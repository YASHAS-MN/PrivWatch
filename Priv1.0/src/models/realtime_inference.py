import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import deque
import os

# 1. Hardware Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Engine starting on: {device}")

# 2. Model Definitions (We bring the LSTM class here to avoid complex imports)
class RawLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=64, num_layers=1):
        super(RawLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.sigmoid(self.fc(lstm_out[:, -1, :]))

def get_resnet_extractor():
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    modules = list(resnet.children())[:-1]
    extractor = torch.nn.Sequential(*modules).to(device)
    extractor.eval()
    return extractor

# 3. The Real-Time Engine
def run_realtime_system(video_path):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_weights", "raw_lstm_model.pth")
    
    # Load Models
    print("Loading models into VRAM...")
    feature_extractor = get_resnet_extractor()
    lstm_model = RawLSTM().to(device)
    lstm_model.load_state_dict(torch.load(MODEL_PATH))
    lstm_model.eval() # CRITICAL: Set to evaluation mode
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize our sliding window (The Conveyor Belt)
    SEQUENCE_LENGTH = 16
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # Open Video Stream (Use 0 for webcam, or path for video file)
    cap = cv2.VideoCapture(video_path)
    
    print("System Online. Press 'q' to quit.")
    
    frame_skip = 2 # Process every 3rd frame to simulate training data spread and save FPS
    frame_count = 0
    current_status = "Gathering Data..."
    status_color = (255, 255, 0) # Cyan
    
    with torch.no_grad(): # No training allowed here, saves massive VRAM
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # We skip frames to match the "spread" we had in training
            if frame_count % frame_skip == 0:
                # 1. Convert cv2 frame (BGR) to PIL Image (RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # 2. Preprocess and get ResNet features
                input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                features = feature_extractor(input_tensor).squeeze().cpu().numpy()
                
                # 3. Add to our sliding window
                frame_buffer.append(features)
                
                # 4. If window is full, run LSTM Prediction!
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    # Shape: (1 Batch, 16 Frames, 2048 Features)
                    sequence_tensor = torch.FloatTensor(np.array(frame_buffer)).unsqueeze(0).to(device)
                    
                    prediction = lstm_model(sequence_tensor).item()
                    
                    if prediction > 0.5:
                        current_status = f"VIOLENCE DETECTED ({prediction*100:.1f}%)"
                        status_color = (0, 0, 255) # Red in BGR
                    else:
                        current_status = f"NORMAL ({100 - prediction*100:.1f}%)"
                        status_color = (0, 255, 0) # Green in BGR
            
            # Display the result on the video
            display_frame = cv2.resize(frame, (800, 600))
            cv2.putText(display_frame, current_status, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
            
            cv2.imshow("PrivWatch - RAW Pipeline Engine", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Let's test it on one of your raw violence videos!
    # Update this path to match an actual file name in your violence folder
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    test_video = os.path.join(BASE_DIR, "data", "raw_videos", "violence", "fi1_xvid.avi") 
    
    # Note: Check your violence folder and replace 'fi1_xvid.avi' with a file you actually have.
    
    run_realtime_system(test_video)