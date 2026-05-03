import os
import numpy as np
import random

def create_synthetic_normal(sequence):
    """Adds subtle spatial jitter to simulate different body types / slight camera shifts."""
    noise = np.random.normal(0, 0.02, sequence.shape) # 2% variance
    # Don't add noise to visibility scores (every 4th element)
    visibility_mask = np.ones(sequence.shape)
    visibility_mask[:, 3::4] = 0 
    return np.clip(sequence + (noise * visibility_mask), 0, 1)

def create_synthetic_collapse(sequence):
    """Forces a normal walking sequence to collapse mathematically."""
    collapse_seq = np.copy(sequence)
    fall_start = random.randint(3, 7) # Start fall randomly between frame 3 and 7
    
    for frame_idx in range(fall_start, 16):
        fall_progress = (frame_idx - fall_start) / (15 - fall_start)
        gravity_factor = fall_progress ** 2 # Exponential drop
        
        for joint_idx in range(33):
            y_pos = (joint_idx * 4) + 1
            z_pos = (joint_idx * 4) + 2
            
            if joint_idx < 25: # Upper body drops fast
                current_y = collapse_seq[frame_idx, y_pos]
                # Push Y towards 1.0 (bottom of screen)
                collapse_seq[frame_idx, y_pos] = min(current_y + (0.5 * gravity_factor), 1.0)
                # Shift Z to simulate falling forward/backward
                collapse_seq[frame_idx, z_pos] += (0.2 * gravity_factor)
                
    return collapse_seq

def augment_data(target_count=100):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    KP_NORMAL = os.path.join(BASE_DIR, "data", "keypoints", "normal")
    KP_COLLAPSE = os.path.join(BASE_DIR, "data", "keypoints", "collapse")
    
    # 1. Augment Normal Data
    normal_files = [f for f in os.listdir(KP_NORMAL) if f.endswith('.npy') and not f.startswith('synth_')]
    current_normal = len(normal_files)
    needed_normal = target_count - current_normal
    
    print(f"Generating {needed_normal} synthetic Normal sequences...")
    for i in range(needed_normal):
        # Pick a random real normal sequence
        base_seq = np.load(os.path.join(KP_NORMAL, random.choice(normal_files)))
        synth_seq = create_synthetic_normal(base_seq)
        np.save(os.path.join(KP_NORMAL, f"synth_normal_{i}.npy"), synth_seq)

    # 2. Augment Collapse Data
    collapse_files = [f for f in os.listdir(KP_COLLAPSE) if f.endswith('.npy') and not f.startswith('synth_')]
    current_collapse = len(collapse_files)
    needed_collapse = target_count - current_collapse
    
    print(f"Generating {needed_collapse} synthetic Collapse sequences...")
    for i in range(needed_collapse):
        # Pick a random normal sequence and FORCE it to collapse
        base_seq = np.load(os.path.join(KP_NORMAL, random.choice(normal_files)))
        synth_seq = create_synthetic_collapse(base_seq)
        np.save(os.path.join(KP_COLLAPSE, f"synth_collapse_{i}.npy"), synth_seq)
        
    print(f"\nAugmentation Complete! Both classes now have {target_count} samples.")

if __name__ == "__main__":
    augment_data(target_count=100)