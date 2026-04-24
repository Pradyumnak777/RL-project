import torch
from data_utils import pointStateProducer
import os
from collections import deque
import random
import numpy as np
import torch.optim as optim
from model import DQN
import torch.nn.functional as F

BATCH_SIZE = 128
GAMMA = 0.95           # Discount factor for future rewards
LR = 1e-4              # Learning rate
TARGET_UPDATE = 20000   # How many steps before syncing Policy -> Target
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_dim=5, action_dim=3).to(device)
target_net = DQN(state_dim=5, action_dim=3).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # Target net never trains via backprop, only copies weights after x steps..
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # 1. Sample a random batch of transitions
    transitions = random.sample(memory.buffer, BATCH_SIZE)
    
    # Zip the batch into separate tensors
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

    state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(reward_batch).to(device)
    done_batch = torch.FloatTensor(done_batch).to(device)

    # 2. Calculate Q(s, a) using Policy Net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 3. Calculate the Target Q-value using Target Net
    # V(s') = max_a Q(s', a)
    with torch.no_grad():
        next_states_non_final = torch.FloatTensor(np.array([
            ns if ns is not None else np.zeros(5) for ns in next_state_batch
        ])).to(device)
        
        next_state_values = target_net(next_states_non_final).max(1)[0]
        
        # Bellman Equation: Q_target = r + gamma * max(Q_next) * (1 - done)
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    # 4. Compute Huber Loss (more robust to outliers than MSE)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)) #diff. between TD target and pred.

    # 5. Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to prevent "exploding gradients" in RL
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

def get_dqn_action(state_vector, can_reanchor=True):
    # Convert numpy array to torch tensor and add batch dimension
    state_t = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get Q-values from the Policy Net
        q_values = policy_net(state_t)
        
        if not can_reanchor:
            # Set the Q-value for action 2 to negative infinity
            # so argmax will NEVER pick it.
            q_values[0, 2] = -float('inf')
        
        # Pick the action with the highest expected reward
        return torch.argmax(q_values).item()
    
#for soft update

TAU = 0.005 # Target network update rate

def soft_update(target_net, policy_net, tau):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def __len__(self):
        return len(self.buffer)


train_dir = "UCF_Rep/train"
memory = ReplayBuffer(capacity=100000)

epsilon = 1.0          # Start with 100% random exploration
epsilon_min = 0.05      # Never stop exploring completely
epsilon_decay = 0.99  # Multiply epsilon by this after every video
global_step = 0       # Tracks total actions taken across all videos


# def get_dqn_action(state_vector):
#     # For now, just pretend the DQN is guessing
#     return random.choice([0, 1, 2]) #0,1,2 are our action choices..

'''
run a loop over the training folder/training videos
'''
stats = {
    "avg_rewards": [],
    "avg_losses": [],
    "epsilon_history": [],
    "survival_rates": [],          # NEW: Tracks point survival percentage
    "action_distributions": []     # NEW: Tracks how often each action is picked
}

for vid_name in os.listdir(train_dir):
    # vid_path = os.path.join(train_dir, vid_name)
    
    print(f"Processing Video: {vid_name}")
    # Initialize the environment for THIS video
    producer = pointStateProducer(vid_name)
    
    # True = Alive, False = Killed by Agent or lost by OpenCV
    active_points = np.ones(producer.num_points, dtype=bool)
    
    video_rewards = []
    video_losses = []
    video_actions = {0: 0, 1: 0, 2: 0} # NEW: Action counters for this video
    current_loss = None
    
    for t in range(producer.num_frames - 1):
        # Loop through every point in the current frame
        for p_idx in range(producer.num_points):
            # Skip if the point was previously killed
            if not active_points[p_idx]:
                continue
            # 1. GET CURRENT STATE (s)
            state = producer.get_state(p_idx, t) #[dx, dy, similarity] ..
            
            if state is None:
                active_points[p_idx] = False
                continue
            
            #check budget for re-anchoring..
            can_reanchor = producer.reanchor_counts[p_idx] < producer.max_reanchors
            
            # 2. CHOOSE ACTION (Epsilon-Greedy)
            if random.random() < epsilon:
                if can_reanchor:
                    action = random.choice([0, 1, 2]) # Full exploration
                else:
                    action = random.choice([0, 1])    # Restricted: Only Kill or Keep, not allowing "re-anchor"
            else:
                # MUST pass the flag to the policy network
                action = get_dqn_action(state, can_reanchor=can_reanchor)
                
            video_actions[action] += 1 # NEW: Log the chosen action
            
            # 3. EXECUTE ACTION
            if action == 0: # KILL
                active_points[p_idx] = False # Mark as dead for future frames
                
            elif action == 2: # RE-ANCHOR
                producer.update_anchor(p_idx, t) #new frame is set as anchor..
                
            # 4. GET REWARD (r)
            reward = producer.get_reward(p_idx, t, action)
            video_rewards.append(reward) # Collect reward
            
            # 5. GET NEXT STATE (s')
            # If we killed it, there is no next state
            if action == 0:
                next_state = None
                done = True
            else:
                next_state = producer.get_state(p_idx, t + 1)
                # It's 'done' if the next state is None (OpenCV loses it next frame)
                done = (next_state is None) 
                if done:
                    active_points[p_idx] = False

            # 6. STORE IN MEMORY
            memory.push(state, action, reward, next_state, done)
            # This is the 'Robbins-Monro' step: correcting the prediction toward the target
            # current_loss = optimize_model()
            global_step += 1
            
            if global_step % 16 == 0: 
                current_loss = optimize_model()
                
                
            if current_loss is not None:
                video_losses.append(current_loss)
            
            #step additions..
            # if global_step % 1000 == 0:
            #     epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # if global_step % TARGET_UPDATE == 0:
            #     target_net.load_state_dict(policy_net.state_dict())
            #     print(f" [Sync] Target Network Updated at step {global_step}")
            
            if global_step % 2000 == 0:
                soft_update(target_net, policy_net, TAU)
                print(f" [Sync] Target Network Soft Updated at step {global_step}")
                
    avg_v_reward = np.mean(video_rewards) if video_rewards else 0
    '''
    #NOTE: its the mean of those (num_of_points * num_of frames) rewards, per step, for all points, in a video..
    '''
    avg_v_loss = np.mean(video_losses) if video_losses else 0
    # NEW: Calculate how many points survived to the end of the video
    survival_rate = np.sum(active_points) / producer.num_points if producer.num_points > 0 else 0 
    
    stats["avg_rewards"].append(avg_v_reward)
    stats["avg_losses"].append(avg_v_loss)
    stats["epsilon_history"].append(epsilon)
    stats["survival_rates"].append(survival_rate)      # NEW
    stats["action_distributions"].append(video_actions) # NEW

    print(f"Done: {vid_name:20} | Rwd: {avg_v_reward:6.2f} | Loss: {avg_v_loss:6.4f} | Eps: {epsilon:.2f} | Surv: {survival_rate:.2f}")

    #decaying epsilon..
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    
import json
with open("training_stats.json", "w") as f:
    json.dump(stats, f)

model_save_path = "point_tracker_dqn.pth"
torch.save(policy_net.state_dict(), model_save_path)
print(f"Training Complete! Model saved to {model_save_path}")