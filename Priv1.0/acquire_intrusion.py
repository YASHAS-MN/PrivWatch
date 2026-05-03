import os
import shutil
import glob

os.environ['KAGGLE_USERNAME'] = "yashasmn10" 
os.environ['KAGGLE_KEY'] = "eca568e32f8db92157083e31d3e2264f"

from kaggle.api.kaggle_api_extended import KaggleApi

def acquire_intrusion():
    api = KaggleApi()
    api.authenticate()
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dir_intrusion = os.path.join(base_dir, "data", "raw_videos", "intrusion")
    temp_dir = os.path.join(base_dir, "temp_intrusion_zip")
    
    os.makedirs(dir_intrusion, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. dheerajperumandla/anomaly-detection-videos: An unlocked CCTV anomaly subset
    # 2. straysheep/ucf-crime-reduced: Unlocked subset
    datasets = [
        "dheerajperumandla/anomaly-detection-videos",
        "straysheep/ucf-crime-reduced"
    ]
    
    keywords = ['burglary', 'robbery', 'vandalism', 'stealing', 'trespass', 'intrusion']
    
    count_intrusion = 0
    target = 50
    
    for dataset_ref in datasets:
        if count_intrusion >= target: 
            break
            
        print(f"\n📥 Bulk downloading Intrusion CCTV data from {dataset_ref}...")
        try:
            api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
            all_videos = glob.glob(os.path.join(temp_dir, '**', '*.*'), recursive=True)
            
            for vid in all_videos:
                if count_intrusion >= target: 
                    break
                    
                filename = os.path.basename(vid).lower()
                filepath = vid.lower()
                
                if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # Hunt for spatial anomaly keywords
                    if any(kw in filepath for kw in keywords):
                        ext = filename.split('.')[-1]
                        new_filename = f"intrusion_cctv_{count_intrusion}.{ext}"
                        
                        shutil.copy(vid, os.path.join(dir_intrusion, new_filename))
                        count_intrusion += 1
            
            print(f"🧹 Cleaning up temp files for {dataset_ref}...")
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"❌ Failed to process {dataset_ref}: {e}")
            
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n✅ INTRUSION ACQUISITION COMPLETE: {count_intrusion}/{target} videos downloaded.")

if __name__ == "__main__":
    acquire_intrusion()