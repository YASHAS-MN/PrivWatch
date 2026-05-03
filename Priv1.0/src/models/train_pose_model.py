import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Detect GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Initializing PrivWatch 4-Class Core on device: {device}")

# --- 1. DATASET HANDLER ---
class PoseFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        # Multi-class requires LongTensor
        self.labels = torch.LongTensor(labels) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_4class_data(base_dir):
    features, labels = [], []
    # THE MASTER 4-CLASS TAXONOMY
    classes = {'normal': 0, 'fight': 1, 'collapse': 2, 'intrusion': 3}
    
    for cls_name, label in classes.items():
        folder = os.path.join(base_dir, "data", "keypoints", cls_name)
        if not os.path.exists(folder):
            print(f"⚠️ Warning: Folder not found for {cls_name}")
            continue
            
        for file in os.listdir(folder):
            if file.endswith('.npy'):
                seq = np.load(os.path.join(folder, file))
                features.append(seq)
                labels.append(label)
                
    return np.array(features), np.array(labels)

# --- 2. THE MULTI-CLASS LSTM ARCHITECTURE ---
class PoseLSTM(nn.Module):
    # input_size=132 (33 joints * 4 coords: x, y, z, visibility)
    def __init__(self, input_size=132, hidden_size=64, num_layers=2, num_classes=4):
        super(PoseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Dropout (0.2) prevents overfitting on our synthetic/MoCap data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 4 Output Neurons for 4 Classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, sequence_length=16, features=132)
        lstm_out, _ = self.lstm(x)
        
        # Extract the final frame's output from the 16-frame sequence
        last_frame_out = lstm_out[:, -1, :]
        
        # Raw logits for CrossEntropyLoss
        out = self.fc(last_frame_out)
        return out

# --- 3. THE TRAINING LOOP ---
def train_master_model():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_weights")
    os.makedirs(MODEL_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(MODEL_DIR, "pose_lstm_4class.pth")
    
    print("📊 Loading 4-Class Dataset...")
    X, y = load_4class_data(BASE_DIR)
    print(f"Total sequences loaded: {len(y)} (Expected: 600)")
    
    # Perfectly balanced train/val split using stratify
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(PoseFeatureDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(PoseFeatureDataset(X_val, y_val), batch_size=16, shuffle=False)
    
    model = PoseLSTM().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50 # Bumped to 50 to let the model learn the complex Intrusion patterns
    best_val_acc = 0.0
    
    print("\n🔥 Commencing Neural Network Training 🔥")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # The prediction is the index of the highest firing neuron
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Print every 5 epochs to keep terminal clean
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            
    print(f"\n🏆 Training Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"💾 Weights saved securely to: {SAVE_PATH}")

if __name__ == "__main__":
    train_master_model()