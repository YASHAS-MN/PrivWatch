import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training Baseline Engine on device: {device}")

class BaselineDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_baseline_data(normal_dir, collapse_dir):
    """Loads Normal (0) and Collapse (1) data."""
    features, labels = [], []
    
    for file in os.listdir(normal_dir):
        if file.endswith('.npy'):
            features.append(np.load(os.path.join(normal_dir, file)))
            labels.append(0) # Normal is 0
            
    for file in os.listdir(collapse_dir):
        if file.endswith('.npy'):
            features.append(np.load(os.path.join(collapse_dir, file)))
            labels.append(1) # Collapse is 1
            
    return np.array(features), np.array(labels)

class PoseLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_layers=1):
        super(PoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.sigmoid(self.fc(lstm_out[:, -1, :]))

def train():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    KP_NORMAL = os.path.join(BASE_DIR, "data", "keypoints", "normal")
    KP_COLLAPSE = os.path.join(BASE_DIR, "data", "keypoints", "collapse")
    
    print("Loading Baseline Dataset...")
    X, y = load_baseline_data(KP_NORMAL, KP_COLLAPSE)
    
    print(f"Total Samples -> Normal: {len(y[y==0])}, Collapse: {len(y[y==1])}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(BaselineDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(BaselineDataset(X_val, y_val), batch_size=16, shuffle=False)
    
    model = PoseLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 30
    best_acc = 0.0
    
    print("Starting Baseline Training...")
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
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%")
        
    print("\nPhase 1 Baseline Complete!")

if __name__ == "__main__":
    train()