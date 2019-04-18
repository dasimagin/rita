import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from curiosity.cnn_features import SimpleConvNet

class CuriosityRewarder(nn.Module):
    def __init__(self, observation_shape, n_actions):
        super(CuriosityRewarder, self).__init__()
        self.n_actions = n_actions
        self.conv_features = SimpleConvNet(observation_shape)
        self.mean = 0
        self.mean_sq = 1
        
        self.fc1 = nn.Linear(self.conv_features.cnn_output_size + n_actions, 300)
        self.fc2 = nn.Linear(300 + n_actions, 800)
        self.fc3 = nn.Linear(800 + n_actions, self.conv_features.cnn_output_size)
        
    def forward(self, state, action_oh):
        with torch.no_grad():
            conv_state_features = self.conv_features(state)
        x = torch.cat([conv_state_features, action_oh], dim=1)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, action_oh], dim=1)
        x = F.relu(self.fc2(x))
        x = torch.cat([x, action_oh], dim=1)
        x = F.relu(self.fc3(x))
        return x
        
    def get_reward(self, state, action, next_state):
        action_oh = torch.zeros((state.shape[0], self.n_actions))
        action_oh[range(state.shape[0]), action] = 1
        predicted_next_state = self.forward(state, action_oh)
        with torch.no_grad():
            real_next_state = self.conv_features(next_state)
        loss = torch.sum((predicted_next_state - real_next_state.detach())**2, dim=1)
        alpha = min(0.5, 0.01 * state.shape[0])
        self.mean = (1 - alpha) * self.mean + alpha * loss.detach().numpy().mean()
        self.mean_sq = (1 - alpha) * self.mean_sq + alpha * (loss.detach().numpy()**2).mean()
        var = self.mean_sq - self.mean**2
        return (loss - self.mean) / (var**0.5)
        
        