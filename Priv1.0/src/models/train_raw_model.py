import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Device configuration - will use your RTX 3050 now!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# 1. Dataset Loader
class VideoFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1) # Shape (N, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(violence_dir, normal_dir):
    """Loads the .npy feature files we generated with ResNet."""
    features = []
    labels = []
    
    # Load Violence (Label 1)
    for file in os.listdir(violence_dir):
        if file.endswith('.npy'):
            features.append(np.load(os.path.join(violence_dir, file)))
            labels.append(1)
            
    # Load Normal (Label 0)
    for file in os.listdir(normal_dir):
        if file.endswith('.npy'):
            features.append(np.load(os.path.join(normal_dir, file)))
            labels.append(0)
            
    return np.array(features), np.array(labels)

# 2. LSTM Architecture
class RawLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=64, num_layers=1):
        super(RawLSTM, self).__init__()
        # The LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer to map LSTM output to a single probability (0 to 1)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, 16 frames, 2048 features)
        lstm_out, (hn, cn) = self.lstm(x)
        # We only care about the output of the LAST frame in the sequence to make a decision
        last_frame_out = lstm_out[:, -1, :] 
        
        out = self.fc(last_frame_out)
        return self.sigmoid(out)

# 3. Training Loop
def train_model():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    FEAT_VIOLENCE = os.path.join(BASE_DIR, "data", "features", "violence")
    FEAT_NORMAL = os.path.join(BASE_DIR, "data", "features", "normal")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "saved_weights", "raw_lstm_model.pth")
    
    print("Loading data...")
    X, y = load_data(FEAT_VIOLENCE, FEAT_NORMAL)
    
    # Split into 80% training, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = VideoFeatureDataset(X_train, y_train)
    val_dataset = VideoFeatureDataset(X_val, y_val)
    
    # Small batch size to ensure no VRAM overloads
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = RawLSTM().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy for 0/1 classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    best_val_acc = 0.0
    
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # Convert probabilities to 0 or 1
                predicted = (outputs > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()