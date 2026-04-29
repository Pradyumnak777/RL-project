import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Use the 'Agg' backend so matplotlib doesn't try to open a window
# (Essential for working on a cluster via SSH)
import matplotlib
matplotlib.use('Agg')

def moving_average(data, window_size):
    if len(data) < window_size or window_size == 0:
        return data 
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 1. Load the Data
file_path = "training_stats_new.json" if os.path.exists("training_stats_new.json") else "training_stats.json"

try:
    with open(file_path, "r") as f:
        stats = json.load(f)
    print(f"Successfully loaded data from {file_path}")
except FileNotFoundError:
    print(f"Error: {file_path} not found. Run training first!")
    exit()

# Extract Data
episode_rewards = stats.get("episode_rewards", [])
episode_lengths = stats.get("episode_lengths", [])

avg_rewards = stats.get("avg_rewards", [])
losses = stats.get("avg_losses", [])
epsilons = stats.get("epsilon_history", [])
survival_rates = stats.get("survival_rates", [])
action_dists = stats.get("action_distributions", [])

if not avg_rewards:
    print("Error: JSON is empty or missing video-level data.")
    exit()

# Set up axes and window sizes
videos = np.arange(len(avg_rewards))
episodes = np.arange(len(episode_rewards))

vid_window = min(20, len(avg_rewards))
# Use a much larger window for episodes because there are hundreds of thousands of them
ep_window = min(1000, len(episode_rewards)) if len(episode_rewards) > 0 else 0

# Process Action Distributions
perc_kill, perc_keep, perc_reanchor = [], [], []
for dist in action_dists:
    total = dist.get("0", 0) + dist.get("1", 0) + dist.get("2", 0)
    if total == 0:
        perc_kill.append(0); perc_keep.append(0); perc_reanchor.append(0)
    else:
        perc_kill.append(dist.get("0", 0) / total * 100)
        perc_keep.append(dist.get("1", 0) / total * 100)
        perc_reanchor.append(dist.get("2", 0) / total * 100)

# --- PLOT 1: EPISODE REWARDS (NEW) ---
if len(episode_rewards) > 0:
    plt.figure(figsize=(12, 6))
    # Use a faint scatter for raw data to avoid a solid block of color
    plt.scatter(episodes, episode_rewards, alpha=0.05, color='cyan', s=1, label='Raw Episode Reward')
    if ep_window > 0:
        plt.plot(np.arange(ep_window-1, len(episode_rewards)), moving_average(episode_rewards, ep_window), 
                 color='blue', linewidth=2, label=f'{ep_window}-Ep Avg')
    plt.title("Individual Point Lifecycles: Reward Evolution", fontsize=14)
    plt.xlabel("Episodes (Individual Points Processed)")
    plt.ylabel("Total Accumulated Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plot_episode_rewards.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot_episode_rewards.png")

# --- PLOT 2: EPISODE LENGTHS / SURVIVAL (NEW) ---
if len(episode_lengths) > 0:
    plt.figure(figsize=(12, 6))
    plt.scatter(episodes, episode_lengths, alpha=0.05, color='thistle', s=1, label='Raw Episode Length')
    if ep_window > 0:
        plt.plot(np.arange(ep_window-1, len(episode_lengths)), moving_average(episode_lengths, ep_window), 
                 color='purple', linewidth=2, label=f'{ep_window}-Ep Avg')
    plt.title("Individual Point Lifecycles: Lifespan Evolution", fontsize=14)
    plt.xlabel("Episodes (Individual Points Processed)")
    plt.ylabel("Frames Survived")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plot_episode_lengths.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: plot_episode_lengths.png")

# --- PLOT 3: VIDEO-LEVEL REWARDS ---
plt.figure(figsize=(10, 6))
plt.plot(videos, avg_rewards, alpha=0.3, color='blue', label='Raw Video Avg Reward')
if vid_window > 0:
    plt.plot(np.arange(vid_window-1, len(avg_rewards)), moving_average(avg_rewards, vid_window), 
             color='darkblue', linewidth=2, label=f'{vid_window}-Vid Avg')
plt.title("DQN Video-Level Reward Convergence", fontsize=14)
plt.xlabel("Videos Processed")
plt.ylabel("Average Reward per Video")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plot_video_rewards.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plot_video_rewards.png")

# --- PLOT 4: LOSS & EPSILON ---
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.plot(videos, losses, alpha=0.3, color='red', label='Raw Loss')
if vid_window > 0:
    ax1.plot(np.arange(vid_window-1, len(losses)), moving_average(losses, vid_window), 
             color='darkred', linewidth=2, label='Avg Loss')
ax2.plot(videos, epsilons, color='green', linewidth=2, linestyle='--', label='Epsilon')
ax1.set_title("Training Loss and Epsilon Decay", fontsize=14)
ax1.set_xlabel("Videos Processed")
ax1.set_ylabel("Huber Loss", color='darkred')
ax2.set_ylabel("Exploration Rate (Epsilon)", color='green')
plt.grid(True, alpha=0.3)
plt.savefig("plot_loss_epsilon.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plot_loss_epsilon.png")

# --- PLOT 5: VIDEO-LEVEL SURVIVAL RATE ---
plt.figure(figsize=(10, 6))
plt.plot(videos, np.array(survival_rates) * 100, alpha=0.3, color='purple', label='Raw Survival %')
if vid_window > 0:
    plt.plot(np.arange(vid_window-1, len(survival_rates)), moving_average(np.array(survival_rates) * 100, vid_window), 
             color='indigo', linewidth=2, label='Avg Survival %')
plt.title("Overall Point Survival Performance", fontsize=14)
plt.xlabel("Videos Processed")
plt.ylabel("Survival Percentage (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plot_video_survival.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plot_video_survival.png")

# --- PLOT 6: ACTION DISTRIBUTION ---
plt.figure(figsize=(10, 6))
if vid_window > 0:
    plt.plot(np.arange(vid_window-1, len(perc_keep)), moving_average(perc_keep, vid_window), color='blue', linewidth=2, label='Keep (1)')
    plt.plot(np.arange(vid_window-1, len(perc_kill)), moving_average(perc_kill, vid_window), color='red', linewidth=2, label='Kill (0)')
    plt.plot(np.arange(vid_window-1, len(perc_reanchor)), moving_average(perc_reanchor, vid_window), color='orange', linewidth=2, label='Re-anchor (2)')
plt.title(f"Policy Evolution - Action Distribution ({vid_window}-Vid Avg)", fontsize=14)
plt.xlabel("Videos Processed")
plt.ylabel("Selection Percentage (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plot_actions.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plot_actions.png")

print("\nAll visualization plots saved successfully.")