import os
import numpy as np
import torch
from tqdm import tqdm  # Great for tracking progress on 400 videos
from func_utils import (
    get_optical_flow, 
    get_dino_sscores, 
    precompute_fb_errors, 
    precompute_neighborhood_errors
)

# Configuration
train_dir = "UCF_Rep/train"
save_dir = "precomputed_data"

# Create the save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")

# Get list of videos
video_files = [f for f in os.listdir(train_dir) if f.endswith('.mp4')]
print(f"Found {len(video_files)} videos to process.")

for vid_name in tqdm(video_files, desc="Precomputing Video Data"):
    vid_path = os.path.join(train_dir, vid_name)
    
    # Define save path (change .mp4 to .npz)
    save_path = os.path.join(save_dir, vid_name.replace('.mp4', '.npz'))
    
    # Check if we already processed this video (Skip if exists)
    if os.path.exists(save_path):
        continue

    try:
        # 1. Physical Tracking & Points
        # all_flows: (T, N, 2), start_points: (N, 2)
        all_flows, start_points = get_optical_flow(vid_path)
        
        # 2. Heavy Vision Transformer Inference (DINOv2)
        # all_descriptors: (T, N, D)
        all_descriptors = get_dino_sscores(vid_path, all_flows, start_points)
        
        # 3. Error Precomputations (Consistency & Neighborhood)
        fb_errors = precompute_fb_errors(vid_path, all_flows, start_points)
        nb_errors = precompute_neighborhood_errors(all_flows, start_points, radius=50)
        
        # 4. Save to Compressed Numpy Archive
        # This keeps the disk usage low on your scratch space
        np.savez_compressed(
            save_path,
            flows=all_flows,
            start_points=start_points,
            descriptors=all_descriptors,
            fb_errors=fb_errors,
            nb_errors=nb_errors
        )
        
    except Exception as e:
        print(f"\nError processing {vid_name}: {e}")
        continue

print(f"\nSuccessfully precomputed all data to {save_dir}/")