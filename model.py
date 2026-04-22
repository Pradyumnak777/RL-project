import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=128):
        """
        DQN for Point Tracking Policy
        state_dim: [dx, dy, similarity]
        action_dim: [Kill, Keep, Re-anchor]
        """
        super(DQN, self).__init__()
        
        # Standard fully connected layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Expects x to be a tensor of shape (batch_size, 3) 
        or (3,) for single inference.
        """
        # Using ReLU for non-linearity
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # No activation on the final layer! 
        # We want the raw Q-values (Expected Future Reward)
        return self.fc3(x)

    def act(self, state, device):
        """
        Helper method for the exploration loop.
        Converts numpy state to tensor and returns the best action index.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.forward(state_t)
            return torch.argmax(q_values).item()