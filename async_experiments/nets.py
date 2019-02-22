import torch
import torch.nn as nn

class ConvQNet(nn.Module):
    def __init__(self, img_shape, n_actions):
        super(ConvQNet, self).__init__()
        self.img_shape = img_shape
        self.n_actions = n_actions
        
        self.cnn_features = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.action_values = nn.Sequential(
            nn.Linear(self.flatten_size(), 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions)
        )
        
        self.state_value = nn.Sequential(
            nn.Linear(self.flatten_size(), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.cnn_features(x)
        x = x.view(x.size(0), -1)
        action_values = self.action_values(x)
        advantage = action_values - action_values.mean()
        state_value = self.state_value(x)
        return state_value + advantage
    
    def flatten_size(self):
        with torch.no_grad():
            return self.cnn_features(torch.zeros(1, *self.img_shape)).view(1, -1).size(1)
