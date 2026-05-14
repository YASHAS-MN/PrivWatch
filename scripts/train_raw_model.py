import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from privwatch.dataset_loader import ClipDataset
from privwatch.model import ActionModel
from privwatch.paths import CLIPS_DATA_DIR

# ================= CONFIG =================
BATCH_SIZE = 2
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Clean retraining experiment started")
print("Using diversified NORMAL dataset")
print("Using device:", DEVICE)

# ================= DATA =================
train_dataset = ClipDataset(CLIPS_DATA_DIR / "train")
val_dataset = ClipDataset(CLIPS_DATA_DIR / "val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = ActionModel().to(DEVICE)
print("Training from scratch: no checkpoint loaded")
best_acc = 0.0

# ================= TRAIN SETUP =================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= TRAIN LOOP =================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, (clips, labels) in enumerate(train_loader):
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # small progress print
        if i % 300 == 0:
            print(f"Epoch {epoch + 1} | Batch {i} | Loss: {loss.item():.4f}")

        if i >= 2500:
            break

    print(f"\nEpoch {epoch + 1} Total Loss: {total_loss:.4f}")

    # ================= VALIDATION =================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(clips)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}\n")
    if val_acc > best_acc:
        best_acc = val_acc
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "val_accuracy": float(val_acc),
            "timestamp": datetime.now().isoformat(),
            "dataset_summary": {
                "train_fight": 2800,
                "train_collapse": 2800,
                "train_normal": 2988,
                "val_fight": 600,
                "val_collapse": 600,
                "val_normal": 626,
            },
        }
        torch.save(checkpoint, "models/best_model.pth")
        print(f"Best model updated at epoch {epoch+1} with accuracy {val_acc:.4f}")
