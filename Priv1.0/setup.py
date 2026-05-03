import os

def create_project_structure():
    directories = [
        "data/raw_videos/violence",
        "data/raw_videos/normal",
        "data/frames/violence",
        "data/frames/normal",
        "data/keypoints/violence",
        "data/keypoints/normal",
        "src/data_prep",
        "src/models",
        "src/utils",
        "models/saved_weights",
        "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create a basic init file in python source folders
        if directory.startswith("src"):
            with open(os.path.join(directory, "__init__.py"), "w") as f:
                pass
                
    print("PrivWatch directory structure created successfully!")

if __name__ == "__main__":
    create_project_structure()