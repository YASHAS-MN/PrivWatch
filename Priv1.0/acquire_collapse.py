import os
import shutil
import glob

os.environ['KAGGLE_USERNAME'] = "yashasmn10" 
os.environ['KAGGLE_KEY'] = "eca568e32f8db92157083e31d3e2264f"

from kaggle.api.kaggle_api_extended import KaggleApi

def acquire_collapse():
    api = KaggleApi()
    api.authenticate()
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dir_collapse = os.path.join(base_dir, "data", "raw_videos", "collapse")
    temp_dir = os.path.join(base_dir, "temp_collapse_zip")
    
    os.makedirs(dir_collapse, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Using the same high-quality biomechanics datasets, but extracting the FALLS.
    datasets = [
        "lockedsoul/human-fall-videos",
        "unidpro/fall-detection"
    ]
    
    count_collapse = 0
    target = 50
    
    for dataset_ref in datasets:
        if count_collapse >= target: 
            break
            
        print(f"\n📥 Bulk downloading Collapse data from {dataset_ref}...")
        try:
            api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
            all_videos = glob.glob(os.path.join(temp_dir, '**', '*.*'), recursive=True)
            
            for vid in all_videos:
                if count_collapse >= target: 
                    break
                    
                filename = os.path.basename(vid).lower()
                filepath = vid.lower()
                
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # EXPLICITLY hunt for "Fall" and explicitly REJECT "ADL" or "Normal"
                    if 'fall' in filepath and 'adl' not in filepath and 'normal' not in filepath:
                        ext = filename.split('.')[-1]
                        new_filename = f"collapse_event_{count_collapse}.{ext}"
                        
                        shutil.copy(vid, os.path.join(dir_collapse, new_filename))
                        count_collapse += 1
            
            print(f"🧹 Cleaning up temp files for {dataset_ref}...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"❌ Failed to process {dataset_ref}: {e}")
            
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n✅ COLLAPSE ACQUISITION COMPLETE: {count_collapse}/{target} videos downloaded.")

if __name__ == "__main__":
    acquire_collapse()