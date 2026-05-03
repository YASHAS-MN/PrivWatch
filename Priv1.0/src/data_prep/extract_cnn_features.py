import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Device configuration (Uses your RTX 3050 if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_feature_extractor():
    """Loads a pre-trained ResNet50 and removes the final classification layer."""
    # Load pretrained ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Remove the fully connected layer (fc) to get raw features (2048 dimensions)
    modules = list(resnet.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules)
    
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval() # Set to evaluation mode (no training)
    return feature_extractor

def extract_features_from_frames(frames_dir, output_dir, feature_extractor):
    """Passes frames through ResNet and saves the feature sequences."""
    print(f"Extracting CNN features from {frames_dir}...")
    
    # Standard ImageNet transformations required by ResNet
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    video_folders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
    
    with torch.no_grad(): # Don't calculate gradients to save VRAM
        for i, folder in enumerate(video_folders):
            folder_path = os.path.join(frames_dir, folder)
            frames = sorted(os.listdir(folder_path))
            
            sequence_features = []
            
            for frame_file in frames:
                frame_path = os.path.join(folder_path, frame_file)
                image = Image.open(frame_path).convert('RGB')
                
                # Preprocess and add batch dimension: (1, 3, 224, 224)
                input_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Forward pass
                feature = feature_extractor(input_tensor)
                
                # Flatten from (1, 2048, 1, 1) to (2048,)
                feature = feature.squeeze().cpu().numpy()
                sequence_features.append(feature)
                
            # Save the sequence of features
            # Shape will be (16 frames, 2048 features)
            np_features = np.array(sequence_features)
            save_path = os.path.join(output_dir, f"{folder}.npy")
            np.save(save_path, np_features)
            
            if i % 50 == 0:
                print(f"Processed {i}/{len(video_folders)} sequences")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    FRAMES_VIOLENCE = os.path.join(BASE_DIR, "data", "frames", "violence")
    FRAMES_NORMAL = os.path.join(BASE_DIR, "data", "frames", "normal")
    
    FEAT_VIOLENCE = os.path.join(BASE_DIR, "data", "features", "violence")
    FEAT_NORMAL = os.path.join(BASE_DIR, "data", "features", "normal")
    
    extractor = get_feature_extractor()
    
    print("--- Starting Violence Class Feature Extraction ---")
    extract_features_from_frames(FRAMES_VIOLENCE, FEAT_VIOLENCE, extractor)
    
    print("\n--- Starting Normal Class Feature Extraction ---")
    extract_features_from_frames(FRAMES_NORMAL, FEAT_NORMAL, extractor)
    
    print("\nResNet50 Feature extraction complete!")