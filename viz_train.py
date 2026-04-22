import json
import matplotlib.pyplot as plt
import numpy as np

# Use the 'Agg' backend so matplotlib doesn't try to open a window
# (Essential for working on a cluster via SSH)
import matplotlib
matplotlib.use('Agg')

def moving_average(data, window_size=20):
    if len(data) < window_size:
        return data 
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 1. Load the Data
try:
    with open("training_stats.json", "r") as f:
        stats = json.load(f)
except FileNotFoundError:
    print("Error: training_stats.json not found. Run training first!")
    exit()

rewards = stats.get("avg_rewards", [])
losses = stats.get("avg_losses", [])
epsilons = stats.get("epsilon_history", [])
survival_rates = stats.get("survival_rates", [])
action_dists = stats.get("action_distributions", [])

if not rewards:
    print("Error: JSON is empty.")
    exit()

videos = np.arange(len(rewards))
window = min(20, len(rewards))

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

# --- PLOT 1: REWARDS ---
plt.figure(figsize=(10, 6))
plt.plot(videos, rewards, alpha=0.3, color='blue', label='Raw Reward')
if len(rewards) >= window:
    plt.plot(np.arange(window-1, len(rewards)), moving_average(rewards, window), 
             color='darkblue', linewidth=2, label=f'{window}-Vid Avg')
plt.title("DQN Reward Convergence", fontsize=14)
plt.xlabel("Videos Processed")
plt.ylabel("Average Reward")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plot_rewards.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plot_rewards.png")

# --- PLOT 2: LOSS & EPSILON ---
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.plot(videos, losses, alpha=0.3, color='red', label='Raw Loss')
if len(losses) >= window:
    ax1.plot(np.arange(window-1, len(losses)), moving_average(losses, window), 
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

# --- PLOT 3: SURVIVAL RATE ---
plt.figure(figsize=(10, 6))
plt.plot(videos, np.array(survival_rates) * 100, alpha=0.3, color='purple', label='Raw Survival %')
if len(survival_rates) >= window:
    plt.plot(np.arange(window-1, len(survival_rates)), moving_average(np.array(survival_rates) * 100, window), 
             color='indigo', linewidth=2, label='Avg Survival %')
plt.title("Point Survival Performance", fontsize=14)
plt.xlabel("Videos Processed")
plt.ylabel("Survival Percentage (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plot_survival.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plot_survival.png")

# --- PLOT 4: ACTION DISTRIBUTION ---
plt.figure(figsize=(10, 6))
if window > 0:
    plt.plot(np.arange(window-1, len(perc_keep)), moving_average(perc_keep, window), color='blue', linewidth=2, label='Keep (1)')
    plt.plot(np.arange(window-1, len(perc_kill)), moving_average(perc_kill, window), color='red', linewidth=2, label='Kill (0)')
    plt.plot(np.arange(window-1, len(perc_reanchor)), moving_average(perc_reanchor, window), color='orange', linewidth=2, label='Re-anchor (2)')
plt.title(f"Policy Evolution - Action Distribution ({window}-Vid Avg)", fontsize=14)
plt.xlabel("Videos Processed")
plt.ylabel("Selection Percentage (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plot_actions.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: plot_actions.png")

print("\nAll visualization plots saved successfully.")