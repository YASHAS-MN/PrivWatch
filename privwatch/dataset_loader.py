import os
import torch
from torch.utils.data import Dataset

CLASS_MAP = {"Normal": 0, "Fight": 1, "Collapse": 2}

class ClipDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for cls in CLASS_MAP:
            class_path = os.path.join(root_dir, cls)
            for file in os.listdir(class_path):
                if file.endswith(".pt"):
                    self.samples.append((
                        os.path.join(class_path, file),
                        CLASS_MAP[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        clip = torch.load(path)  # (T, H, W, C)
        clip = clip.permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(clip.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(clip.device)
        clip = (clip - mean) / std

        return clip, label
