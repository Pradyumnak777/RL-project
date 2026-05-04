import torch
import cv2
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg') # For headless environments
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_utils import pointStateProducer
from model import DQN
from func_utils import get_optical_flow, get_dino_sscores, precompute_fb_errors, precompute_neighborhood_errors

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_dir = "UCF_Rep/val" 
output_vid_dir = "validation_renders"
model_path = "point_tracker_dqn.pth"
testing_dir = "precomputed_data_testing"

# Toggle this if you want to save the annotated .mp4 files (slows down evaluation)
SAVE_RENDERED_VIDEOS = True 

os.makedirs(testing_dir, exist_ok=True)
if SAVE_RENDERED_VIDEOS:
    os.makedirs(output_vid_dir, exist_ok=True)

# 1. Load the Trained Model
print(f"Loading model from {model_path}...")
policy_net = DQN(state_dim=4, action_dim=3).to(device)
policy_net.load_state_dict(torch.load(model_path, map_location=device))
policy_net.eval()

# 2. Setup Statistics Tracking
test_stats = {
    "episode_rewards": [],
    "episode_lengths": [],
    "avg_rewards": [],
    "survival_rates": [],
    "action_distributions": []
}

val_videos = [f for f in os.listdir(val_dir) if f.endswith('.mp4')]
print(f"Found {len(val_videos)} videos in validation set.")

# 3. Main Validation Loop
for vid_name in tqdm(val_videos, desc="Evaluating Validation Set"):
    test_vid_path = os.path.join(val_dir, vid_name)
    npz_name = vid_name.replace('.mp4', '.npz')
    save_path = os.path.join(testing_dir, npz_name)

    # --- Pre-computation ---
    if not os.path.exists(save_path):
        print(f"\nPrecomputing data for {vid_name}...")
        all_flows, start_points = get_optical_flow(test_vid_path)
        all_descriptors = get_dino_sscores(test_vid_path, all_flows, start_points)
        fb_errors = precompute_fb_errors(test_vid_path, all_flows, start_points)
        nb_errors = precompute_neighborhood_errors(all_flows, start_points, radius=50)
        
        np.savez_compressed(
            save_path, flows=all_flows, start_points=start_points,
            descriptors=all_descriptors, fb_errors=fb_errors, nb_errors=nb_errors
        )

    # --- Environment Initialization ---
    producer = pointStateProducer(vid_name, data_dir=testing_dir)
    active_points = np.ones(producer.num_points, dtype=bool)
    curr_positions = producer.start_points.reshape(-1, 2).copy()
    
    # Trackers for this specific video
    point_rewards = np.zeros(producer.num_points) 
    point_lengths = np.zeros(producer.num_points)
    video_actions = {0: 0, 1: 0, 2: 0}

    # --- Video Writer Setup (Optional) ---
    if SAVE_RENDERED_VIDEOS:
        output_vid_path = os.path.join(output_vid_dir, vid_name)
        cap = cv2.VideoCapture(test_vid_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        ret, frame = cap.read()
        if ret: out.write(frame)
    else:
        # If not rendering, we don't need to read the frames with OpenCV
        pass

    # --- Inference Loop ---
    for t in range(producer.num_frames - 1):
        if SAVE_RENDERED_VIDEOS:
            ret, frame = cap.read()
            if not ret: break
            
        curr_positions = curr_positions + np.nan_to_num(producer.all_flows[t])
        frame_actions = np.full(producer.num_points, -1, dtype=int)

        for p_idx in range(producer.num_points):
            if not active_points[p_idx]:
                continue
                
            state = producer.get_state(p_idx, t)
            
            if state is None:
                active_points[p_idx] = False
                frame_actions[p_idx] = 0
                continue
                
            can_reanchor = producer.reanchor_counts[p_idx] < producer.max_reanchors

            # Inference with Masking
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_t)
                if not can_reanchor:
                    q_values[0, 2] = -float('inf')
                action = torch.argmax(q_values).item()
                
            frame_actions[p_idx] = action
            video_actions[action] += 1
            
            # --- GET REWARD FOR EVALUATION ---
            reward = producer.get_reward(p_idx, t, action)
            point_rewards[p_idx] += reward
            point_lengths[p_idx] += 1
                
            # Apply Logic
            if action == 0:   # KILL
                active_points[p_idx] = False
            elif action == 2: # RE-ANCHOR
                producer.update_anchor(p_idx, t)

        # Render Actions
        if SAVE_RENDERED_VIDEOS:
            for p_idx in range(producer.num_points):
                x, y = curr_positions[p_idx]
                if np.isnan(x) or np.isnan(y): continue

                act = frame_actions[p_idx]
                if act == 1:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
                elif act == 2:
                    cv2.rectangle(frame, (int(x)-3, int(y)-3), (int(x)+3, int(y)+3), (255, 255, 0), -1)
                elif act == 0:
                    cv2.line(frame, (int(x)-4, int(y)-4), (int(x)+4, int(y)+4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
                    cv2.line(frame, (int(x)-4, int(y)+4), (int(x)+4, int(y)-4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            out.write(frame)

    if SAVE_RENDERED_VIDEOS:
        cap.release()
        out.release()

    # --- Aggregate Video Stats ---
    test_stats["episode_rewards"].extend(point_rewards.tolist())
    test_stats["episode_lengths"].extend(point_lengths.tolist())
    
    # Note: Mean across all points in the video (total accumulated / num points)
    avg_v_reward = np.mean(point_rewards)
    survival_rate = np.sum(active_points) / producer.num_points if producer.num_points > 0 else 0 
    
    test_stats["avg_rewards"].append(float(avg_v_reward))
    test_stats["survival_rates"].append(float(survival_rate))
    test_stats["action_distributions"].append(video_actions)

# 4. Save Statistics
with open("validation_stats.json", "w") as f:
    json.dump(test_stats, f)
print("\nValidation Complete! Stats saved to validation_stats.json")

# 5. Generate Evaluation Plots
print("Generating evaluation plots...")

def moving_average(data, window_size):
    if len(data) < window_size or window_size == 0:
        return data 
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

videos = np.arange(len(test_stats["avg_rewards"]))
episodes = np.arange(len(test_stats["episode_rewards"]))
vid_window = min(5, len(videos))

# Plot 1: Video-Level Rewards
plt.figure(figsize=(10, 6))
plt.plot(videos, test_stats["avg_rewards"], alpha=0.5, color='blue', label='Val Video Avg Reward')
if vid_window > 0:
    plt.plot(np.arange(vid_window-1, len(test_stats["avg_rewards"])), 
             moving_average(test_stats["avg_rewards"], vid_window), 
             color='darkblue', linewidth=2, label=f'{vid_window}-Vid Avg')
plt.title("Evaluation: Average Reward per Video", fontsize=14)
plt.xlabel("Validation Videos")
plt.ylabel("Reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("val_plot_video_rewards.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Survival Rate
plt.figure(figsize=(10, 6))
plt.plot(videos, np.array(test_stats["survival_rates"]) * 100, alpha=0.5, color='purple', label='Val Survival %')
if vid_window > 0:
    plt.plot(np.arange(vid_window-1, len(test_stats["survival_rates"])), 
             moving_average(np.array(test_stats["survival_rates"]) * 100, vid_window), 
             color='indigo', linewidth=2, label='Avg Survival %')
plt.title("Evaluation: Point Survival Performance", fontsize=14)
plt.xlabel("Validation Videos")
plt.ylabel("Survival Percentage (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("val_plot_survival.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Action Distribution
perc_kill, perc_keep, perc_reanchor = [], [], []
for dist in test_stats["action_distributions"]:
    total = dist.get("0", 0) + dist.get("1", 0) + dist.get("2", 0)
    if total == 0:
        perc_kill.append(0); perc_keep.append(0); perc_reanchor.append(0)
    else:
        perc_kill.append(dist.get("0", 0) / total * 100)
        perc_keep.append(dist.get("1", 0) / total * 100)
        perc_reanchor.append(dist.get("2", 0) / total * 100)

plt.figure(figsize=(10, 6))
if vid_window > 0:
    plt.plot(np.arange(vid_window-1, len(perc_keep)), moving_average(perc_keep, vid_window), color='blue', linewidth=2, label='Keep (1)')
    plt.plot(np.arange(vid_window-1, len(perc_kill)), moving_average(perc_kill, vid_window), color='red', linewidth=2, label='Kill (0)')
    plt.plot(np.arange(vid_window-1, len(perc_reanchor)), moving_average(perc_reanchor, vid_window), color='orange', linewidth=2, label='Re-anchor (2)')
plt.title(f"Evaluation: Action Distribution ({vid_window}-Vid Avg)", fontsize=14)
plt.xlabel("Validation Videos")
plt.ylabel("Selection Percentage (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("val_plot_actions.png", dpi=300, bbox_inches='tight')
plt.close()

print("Evaluation plots saved successfully.")