import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ActionLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=64, num_layers=2, num_classes=4):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class PoseDataset(Dataset):
    def __init__(self, base_path):
        self.data, self.labels = [], []
        classes = {'normal': 0, 'fight': 1, 'collapse': 2, 'intrusion': 3}
        for cls_name, label in classes.items():
            path = os.path.join(base_path, cls_name)
            for f in os.listdir(path):
                if f.endswith('.npy'):
                    arr = np.load(os.path.join(path, f)).astype(np.float32)
                    # NO HACKS HERE. Data is now naturally (16, 51)
                    self.data.append(arr)
                    self.labels.append(label)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): 
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PoseDataset(r'C:\Prototypes\Priv2.0\data\keypoints')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = ActionLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🧠 Training on {len(dataset)} clean samples...")
    for epoch in range(60):
        for poses, labels in loader:
            poses, labels = poses.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(poses)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1} done.")

    torch.save(model.state_dict(), 'models/action_model.pth')
    print("🏁 SUCCESS: Clean action_model.pth generated.")

if __name__ == "__main__":
    train()