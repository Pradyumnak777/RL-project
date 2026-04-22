import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

# simple neural network to act as value function approximator
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=3, action_size=2):
        self.state_size = state_size # [dx, dy, similarity]
        self.action_size = action_size # [0: kill, 1: keep]
        self.memory = deque(maxlen=2000) #2000 experiences..
        self.gamma = 0.95 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, train=True):
        # epsilon-greedy for exploration vs exploitation
        if train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, path):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location="cpu")

        # supports both full checkpoint and raw state_dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "target_model_state_dict" in checkpoint:
                self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            else:
                self.update_target_network()

            if load_optimizer and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "epsilon" in checkpoint:
                self.epsilon = checkpoint["epsilon"]
        else:
            self.model.load_state_dict(checkpoint)
            self.update_target_network()

    def learn(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            
            # bellman equation: Q(s,a) = r + gamma * max(Q(s',a'))
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item() #this is Q learning! as we're taking
                #MAX without calculating action at next state
            
            current_q = self.model(state)[action]
            
            loss = nn.MSELoss()(current_q, torch.tensor(target))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay