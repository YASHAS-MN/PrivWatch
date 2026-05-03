import os
import shutil
import glob

# Verified Credentials
os.environ['KAGGLE_USERNAME'] = "yashasmn10" 
os.environ['KAGGLE_KEY'] = "eca568e32f8db92157083e31d3e2264f"

from kaggle.api.kaggle_api_extended import KaggleApi

def acquire_baseline_normal():
    api = KaggleApi()
    api.authenticate()
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dir_normal = os.path.join(base_dir, "data", "raw_videos", "normal")
    temp_dir = os.path.join(base_dir, "temp_normal_adl_zip")
    
    os.makedirs(dir_normal, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # We use datasets known to contain rich "ADL" (Activities of Daily Living) 
    # This provides the vital "Sitting at desk" and "Idling" variance.
    datasets = [
        "lockedsoul/human-fall-videos",
        "unidpro/fall-detection"
    ]
    
    count_normal = 0
    target = 50
    
    for dataset_ref in datasets:
        if count_normal >= target: 
            break
            
        print(f"\n📥 Bulk downloading ADL Baseline from {dataset_ref}...")
        try:
            api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
            
            all_videos = glob.glob(os.path.join(temp_dir, '**', '*.*'), recursive=True)
            
            for vid in all_videos:
                if count_normal >= target: 
                    break
                    
                filename = os.path.basename(vid).lower()
                filepath = vid.lower()
                
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # We EXPLICITLY hunt for "ADL" (Activities of Daily Living) or "Normal"
                    if 'adl' in filepath or 'normal' in filepath or 'non' in filepath:
                        ext = filename.split('.')[-1]
                        new_filename = f"normal_adl_{count_normal}.{ext}"
                        
                        shutil.copy(vid, os.path.join(dir_normal, new_filename))
                        count_normal += 1
            
            print(f"🧹 Cleaning up temp files for {dataset_ref}...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"❌ Failed to process {dataset_ref}: {e}")
            
    # Final cleanup just in case
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n✅ NORMAL ACQUISITION COMPLETE: {count_normal}/{target} videos downloaded.")

if __name__ == "__main__":
    acquire_baseline_normal()