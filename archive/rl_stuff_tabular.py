import numpy as np
import random
import pickle
import os

class TabularAgent:
    def __init__(self, action_size=2):
        self.action_size = action_size # [0: kill, 1: keep]
        self.gamma = 0.95              # discount rate
        self.lr = 0.1                 # learning rate for tabular
        self.epsilon = 1.0            # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Q-Table dimensions: [dx_bins, dy_bins, sim_bins, actions]
        # We use 10 bins for each dimension = 1,000 possible states
        self.q_table = np.zeros((10, 10, 10, self.action_size))

    def discretize(self, state):
        """Converts continuous [dx, dy, sim] into discrete table indices."""
        dx, dy, sim = state
        
        # Map dx/dy (roughly -10 to 10 pixels) to 0-9 bins
        # We shift by 10 to make it positive, then divide by 2 to get 10 bins
        dx_bin = int(np.clip((dx + 10) / 2, 0, 9))
        dy_bin = int(np.clip((dy + 10) / 2, 0, 9))
        
        # Map similarity (0.0 to 1.0) to 0-9 bins
        sim_bin = int(np.clip(sim * 10, 0, 9))
        
        return (dx_bin, dy_bin, sim_bin)

    def act(self, state, train=True):
        state_idx = self.discretize(state)
        
        # Epsilon-greedy exploration
        if train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: pick the action with the highest value in the table
        return np.argmax(self.q_table[state_idx])

    def learn(self, state, action, reward, next_state):
        """Standard Q-Learning Update: Q(s,a) = Q(s,a) + lr * [R + gamma * max Q(s',a') - Q(s,a)]"""
        s_idx = self.discretize(state)
        ns_idx = self.discretize(next_state)
        
        # The Bellman Equation logic
        best_next_q = np.max(self.q_table[ns_idx])
        td_target = reward + self.gamma * best_next_q
        
        # Update the specific cell in the Q-table
        current_q = self.q_table[s_idx][action]
        self.q_table[s_idx][action] += self.lr * (td_target - current_q)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']