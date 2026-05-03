import os
import shutil
import glob

# Your verified credentials
os.environ['KAGGLE_USERNAME'] = "yashasmn10" 
os.environ['KAGGLE_KEY'] = "eca568e32f8db92157083e31d3e2264f"

from kaggle.api.kaggle_api_extended import KaggleApi

def download_collapse_bulletproof():
    api = KaggleApi()
    api.authenticate()
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dir_collapse = os.path.join(base_dir, "data", "raw_videos", "collapse")
    temp_dir = os.path.join(base_dir, "temp_collapse_zip")
    
    os.makedirs(dir_collapse, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # We will try these datasets in order until we hit 50 videos
    datasets = [
        "lockedsoul/human-fall-videos",
        "unidpro/fall-detection",
        "soumicksarker/multiple-cameras-fall-dataset"
    ]
    
    count_collapse = 0
    target = 50
    
    for dataset_ref in datasets:
        if count_collapse >= target: 
            break
            
        print(f"\n📥 Bulk downloading {dataset_ref}...")
        try:
            # Download and unzip the entire dataset locally
            api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
            
            # Find all videos recursively in the temp folder
            all_videos = glob.glob(os.path.join(temp_dir, '**', '*.*'), recursive=True)
            
            for vid in all_videos:
                if count_collapse >= target: 
                    break
                    
                filename = os.path.basename(vid).lower()
                filepath = vid.lower()
                
                # Check if it's a valid video format
                if filename.endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    # IMPORTANT: Fall datasets include "ADL" (Activities of Daily Living - normal walking).
                    # We MUST skip ADL videos so we only get pure anomalies.
                    if 'adl' in filepath or 'normal' in filepath:
                        continue
                        
                    # Rename to avoid collisions (01.mp4 from different folders overwriting each other)
                    ext = filename.split('.')[-1]
                    new_filename = f"fall_{count_collapse}.{ext}"
                    
                    shutil.copy(vid, os.path.join(dir_collapse, new_filename))
                    count_collapse += 1
            
            print(f"🧹 Cleaning up temp files for {dataset_ref}...")
            # Clean temp directory before downloading the next dataset to save your SSD
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                    
        except Exception as e:
            print(f"❌ Failed to process {dataset_ref}: {e}")
            
    # Final cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n✅ SUCCESS! Collapse Videos Downloaded: {count_collapse}/{target}")

if __name__ == "__main__":
    download_collapse_bulletproof()