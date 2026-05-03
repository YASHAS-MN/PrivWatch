import os
import numpy as np
import random

def add_kinematic_jitter(sequence, noise_level=0.015):
    """
    Injects Gaussian noise into the X, Y, and Z coordinates.
    Skips the Visibility score (every 4th element in the 132-length array).
    """
    augmented = np.copy(sequence)
    noise = np.random.normal(0, noise_level, augmented.shape)
    
    mask = np.ones(augmented.shape)
    mask[:, 3::4] = 0  # Do not corrupt visibility scores
    
    augmented += (noise * mask)
    augmented = np.clip(augmented, -1.0, 1.0)
    return augmented

def execute_approach_c():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    # ALL 4 CLASSES
    classes = ["normal", "fight", "collapse", "intrusion"]
    
    for cls in classes:
        kp_dir = os.path.join(base_dir, "data", "keypoints", cls)
        
        if not os.path.exists(kp_dir):
            continue
            
        real_files = [f for f in os.listdir(kp_dir) if f.endswith('.npy') and not f.startswith('synth_')]
        current_count = len(real_files)
        target_count = 100
        needed = target_count - current_count
        
        if needed <= 0 or current_count == 0:
            print(f"✅ {cls.upper()} skipped (Has {current_count} files).")
            continue
            
        print(f"\n⚙️ Augmenting {cls.upper()}... Generating {needed} synthetic sequences.")
        
        for i in range(needed):
            source_file = random.choice(real_files)
            source_path = os.path.join(kp_dir, source_file)
            
            seq = np.load(source_path)
            synth_seq = add_kinematic_jitter(seq, noise_level=0.02)
            
            save_path = os.path.join(kp_dir, f"synth_aug_{i}_{source_file}")
            np.save(save_path, synth_seq)
            
        final_count = len([f for f in os.listdir(kp_dir) if f.endswith('.npy')])
        print(f"✅ {cls.upper()} augmented! Total sequences: {final_count}")

if __name__ == "__main__":
    print("🚀 INITIATING MATHEMATICAL KINEMATIC AUGMENTATION...")
    execute_approach_c()
    print("\n🎉 PHASE 1 & DATA PIPELINE COMPLETELY FINISHED! READY FOR 4-CLASS LSTM TRAINING.")