import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import os
import mediapipe as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Privacy Engine starting on: {device}")

# 1. Model Definition
class PoseLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_layers=1):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.sigmoid(self.fc(lstm_out[:, -1, :]))

# 2. MediaPipe Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_extractor = mp_pose.Pose(
    static_image_mode=False, # False because this is a video stream
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. The Privacy Real-Time Engine
def run_privacy_system(video_path):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_weights", "pose_lstm_model.pth")
    
    print("Loading Privacy Model into VRAM...")
    lstm_model = PoseLSTM().to(device)
    lstm_model.load_state_dict(torch.load(MODEL_PATH))
    lstm_model.eval()
    
    SEQUENCE_LENGTH = 16
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    cap = cv2.VideoCapture(video_path)
    
    # We want to loop the short video so you can actually watch it work
    print("System Online. Press 'q' to quit.")
    
    frame_skip = 2 
    frame_count = 0
    current_status = "Gathering Kinematic Data..."
    status_color = (255, 255, 0)
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop the video if it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_count += 1
            
            # Create a completely black frame for the "Privacy View"
            privacy_frame = np.zeros(frame.shape, dtype=np.uint8)
            
            if frame_count % frame_skip == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_extractor.process(rgb_frame)
                
                frame_landmarks = []
                if results.pose_landmarks:
                    # Draw the skeleton on the BLACK frame, not the raw frame
                    mp_drawing.draw_landmarks(
                        privacy_frame, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )
                    
                    # Extract the 132 features
                    for lm in results.pose_landmarks.landmark:
                        frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    frame_landmarks = list(np.zeros(132))
                    
                frame_buffer.append(frame_landmarks)
                
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    sequence_tensor = torch.FloatTensor(np.array(frame_buffer)).unsqueeze(0).to(device)
                    prediction = lstm_model(sequence_tensor).item()
                    
                    if prediction > 0.5:
                        current_status = f"VIOLENCE DETECTED ({prediction*100:.1f}%)"
                        status_color = (0, 0, 255) 
                    else:
                        current_status = f"NORMAL ({100 - prediction*100:.1f}%)"
                        status_color = (0, 255, 0) 
            
            # Display ONLY the privacy frame
            display_frame = cv2.resize(privacy_frame, (800, 600))
            cv2.putText(display_frame, current_status, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
            
            cv2.imshow("PrivWatch - PRIVACY Pipeline Engine", display_frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'): # Added slight delay to make it watchable
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # Make sure this points to the exact video you just used!
    test_video = os.path.join(BASE_DIR, "data", "raw_videos", "violence", "fi1_xvid.avi") 
    
    run_privacy_system(test_video)