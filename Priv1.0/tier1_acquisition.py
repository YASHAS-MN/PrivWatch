import os
import shutil
import glob

# Verified Credentials
os.environ['KAGGLE_USERNAME'] = "yashasmn10" 
os.environ['KAGGLE_KEY'] = "eca568e32f8db92157083e31d3e2264f"

from kaggle.api.kaggle_api_extended import KaggleApi

def acquire_scvd_dataset():
    api = KaggleApi()
    api.authenticate()
    
    # The SCVD dataset is 1GB, allowing us to safely use the bulk download method
    dataset_ref = "toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd"
    print(f"\n📥 Bulk downloading {dataset_ref} (Approx 1GB)...")
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dir_normal = os.path.join(base_dir, "data", "raw_videos", "normal")
    dir_fight = os.path.join(base_dir, "data", "raw_videos", "fight")
    temp_dir = os.path.join(base_dir, "temp_scvd_zip")
    
    os.makedirs(dir_normal, exist_ok=True)
    os.makedirs(dir_fight, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # unzip=True prevents the 404 routing error by pulling the whole archive at once
        api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # Scan the temporary folder for all extracted videos
    all_videos = glob.glob(os.path.join(temp_dir, '**', '*.*'), recursive=True)
    
    count_normal, count_fight = 0, 0
    target = 50
    
    print("\n📂 Distributing diverse CCTV sequences...")
    for vid in all_videos:
        if count_normal >= target and count_fight >= target:
            break
            
        filename = os.path.basename(vid).lower()
        filepath = vid.lower()
        
        # Accept standard video formats
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            
            # Non-Violence (Normal Street Activity / Sitting / Walking)
            if "non" in filepath and count_normal < target:
                ext = filename.split('.')[-1]
                shutil.copy(vid, os.path.join(dir_normal, f"normal_cctv_{count_normal}.{ext}"))
                count_normal += 1
                
            # Violence / Weaponized Violence
            elif "violence" in filepath and "non" not in filepath and count_fight < target:
                ext = filename.split('.')[-1]
                shutil.copy(vid, os.path.join(dir_fight, f"fight_cctv_{count_fight}.{ext}"))
                count_fight += 1

    print("🧹 Cleaning up 1GB of temporary files to save SSD space...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n✅ TIER 1 ACQUISITION COMPLETE.")
    print(f"   Normal Videos: {count_normal}/{target}")
    print(f"   Fight Videos: {count_fight}/{target}")

if __name__ == "__main__":
    acquire_scvd_dataset()