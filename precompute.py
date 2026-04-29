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
save_dir = "precomputed_data_old"

# Create the save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")


# Get list of all .mp4 files in train_dir
video_files = [f for f in os.listdir(train_dir) if f.endswith('.mp4')]
print(f"Found {len(video_files)} videos in {train_dir}.")

# Get list of all .npz files in save_dir
precomputed_files = set(f.replace('.npz', '') for f in os.listdir(save_dir) if f.endswith('.npz'))

missing_videos = []

for vid_name in tqdm(video_files, desc="Checking and Precomputing Missing Data"):
    base_name = vid_name.replace('.mp4', '')
    save_path = os.path.join(save_dir, base_name + '.npz')
    vid_path = os.path.join(train_dir, vid_name)

    if base_name in precomputed_files and os.path.exists(save_path):
        continue

    print(f"Precomputing for missing: {vid_name}")
    try:
        all_flows, start_points = get_optical_flow(vid_path)
        all_descriptors = get_dino_sscores(vid_path, all_flows, start_points)
        fb_errors = precompute_fb_errors(vid_path, all_flows, start_points)
        nb_errors = precompute_neighborhood_errors(all_flows, start_points, radius=50)
        np.savez_compressed(
            save_path,
            flows=all_flows,
            start_points=start_points,
            descriptors=all_descriptors,
            fb_errors=fb_errors,
            nb_errors=nb_errors
        )
        missing_videos.append(vid_name)
    except Exception as e:
        print(f"\nError processing {vid_name}: {e}")
        continue

if missing_videos:
    print(f"\nPrecomputed missing data for {len(missing_videos)} videos:")
    for v in missing_videos:
        print(f"  - {v}")
else:
    print("\nAll videos already have precomputed data.")