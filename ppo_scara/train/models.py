# ==================== models.py ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * 1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, states, actions):
        mean, log_std = self.forward(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.v(x).squeeze(-1)