import os
import zipfile

# --- THE AUTHENTICATION BYPASS ---
os.environ['KAGGLE_USERNAME'] = "yashasmn10" 
os.environ['KAGGLE_KEY'] = "eca568e32f8db92157083e31d3e2264f"

from kaggle.api.kaggle_api_extended import KaggleApi

def download_precise_samples(dataset_ref, output_dir, target_count=50, keyword=""):
    api = KaggleApi()
    api.authenticate()
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nScanning Kaggle dataset: {dataset_ref}...")
    
    try:
        files = api.dataset_list_files(dataset_ref).files
    except Exception as e:
        print(f"❌ HTTP ERROR accessing {dataset_ref}. Details: {e}")
        return
    
    # Expanded video extensions
    video_exts = ('.mp4', '.avi', '.mpg', '.mpeg', '.mkv', '.mov')
    valid_files = []
    
    for f in files:
        filepath = str(f).lower()
        
        # BIG FIX: We check the ENTIRE path for the keyword, not just the file name. 
        if filepath.endswith(video_exts) and keyword.lower() in filepath:
            valid_files.append(f)
    
    if not valid_files:
        print(f"❌ No matching videos found containing '{keyword}'.")
        print("   🔍 DEBUG - Available files on server look like this:")
        # Print the first 5 things on the server so we know exactly what we are dealing with
        for sample in files[:5]:
            print(f"   - {str(sample)}")
        return

    print(f"Found {len(valid_files)} matching files. Downloading first {target_count}...")
    
    downloaded_count = 0
    for f in valid_files:
        if downloaded_count >= target_count:
            break
            
        file_name = str(f)
        print(f"Downloading {downloaded_count + 1}/{target_count}: {file_name.split('/')[-1]}")
        
        try:
            api.dataset_download_file(dataset_ref, file_name=file_name, path=output_dir)
            downloaded_count += 1
        except Exception as e:
            print(f"   Failed to download {file_name}: {e}")
        
    # Clean up any zipped files Kaggle sends
    for item in os.listdir(output_dir):
        if item.endswith('.zip'):
            zip_path = os.path.join(output_dir, item)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                os.remove(zip_path) 
            except:
                pass
            
    print(f"✅ Successfully populated {output_dir} with {downloaded_count} videos.")

if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # 1. NORMAL Class
    RAW_NORMAL = os.path.join(BASE_DIR, "data", "raw_videos", "normal")
    download_precise_samples("yassershrief/hockey-fight-vidoes", RAW_NORMAL, target_count=50, keyword="no")
    
    # 2. FIGHT Class
    RAW_FIGHT = os.path.join(BASE_DIR, "data", "raw_videos", "fight")
    download_precise_samples("yassershrief/hockey-fight-vidoes", RAW_FIGHT, target_count=50, keyword="fi")
    
    # 3. COLLAPSE Class
    RAW_COLLAPSE = os.path.join(BASE_DIR, "data", "raw_videos", "collapse")
    download_precise_samples("sumanpunshi/ur-fall-detection-dataset", RAW_COLLAPSE, target_count=50, keyword="fall")
    
    # 4. INTRUSION Class
    RAW_INTRUSION = os.path.join(BASE_DIR, "data", "raw_videos", "intrusion")
    download_precise_samples("odins0n/ucf-crime-dataset", RAW_INTRUSION, target_count=50, keyword="burglary")
    
    print("\n🎉 Phase 1 (Real Data Acquisition) Complete for all 4 Classes!")