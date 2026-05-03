import os
import shutil
import glob

os.environ['KAGGLE_USERNAME'] = "yashasmn10" 
os.environ['KAGGLE_KEY'] = "eca568e32f8db92157083e31d3e2264f"

from kaggle.api.kaggle_api_extended import KaggleApi

def download_intrusion_bulletproof():
    api = KaggleApi()
    api.authenticate()
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    dir_intrusion = os.path.join(base_dir, "data", "raw_videos", "intrusion")
    temp_dir = os.path.join(base_dir, "temp_intrusion_zip")
    
    os.makedirs(dir_intrusion, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Live Kaggle datasets known to contain actual .mp4/.avi video files for UCF-Crime
    datasets = [
        "yasserhessein/ucf-crime-video-dataset",
        "matthewchin/ucf-crime-dataset",
        "mission-ucf-crime" # Fallback
    ]
    
    # Keywords representing Intrusion / Trespassing in CCTV
    keywords = ['burglary', 'robbery', 'stealing']
    
    count_intrusion = 0
    target = 50
    
    for dataset_ref in datasets:
        if count_intrusion >= target: 
            break
            
        print(f"\n📥 Bulk downloading {dataset_ref} (This might take a moment, these are CCTV videos)...")
        try:
            api.dataset_download_files(dataset_ref, path=temp_dir, unzip=True)
            
            all_videos = glob.glob(os.path.join(temp_dir, '**', '*.*'), recursive=True)
            
            for vid in all_videos:
                if count_intrusion >= target: 
                    break
                    
                filename = os.path.basename(vid).lower()
                filepath = vid.lower()
                
                if filename.endswith(('.avi', '.mp4', '.mkv')):
                    # Check if any of our intrusion keywords are in the file path
                    if any(kw in filepath for kw in keywords):
                        ext = filename.split('.')[-1]
                        new_filename = f"intrusion_{count_intrusion}.{ext}"
                        
                        shutil.copy(vid, os.path.join(dir_intrusion, new_filename))
                        count_intrusion += 1
            
            print(f"🧹 Cleaning up temp files for {dataset_ref}...")
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isfile(item_path): os.remove(item_path)
                elif os.path.isdir(item_path): shutil.rmtree(item_path, ignore_errors=True)
                    
        except Exception as e:
            print(f"❌ Failed to process {dataset_ref}: {e}")
            
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n✅ SUCCESS! Intrusion Videos Downloaded: {count_intrusion}/{target}")

if __name__ == "__main__":
    download_intrusion_bulletproof()