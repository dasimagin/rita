import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):
    def __init__(self, obs_channels, n_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(obs_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 6 * 6, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, n_actions)

        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.critic_linear.weight)
        torch.nn.init.xavier_uniform_(self.actor_linear.weight)

        self.reset_hidden()

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 6 * 6)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))

        return self.critic_linear(self.hx), self.actor_linear(self.hx)
    
    def reset_hidden(self):
        self.cx = torch.zeros(1, 256)
        self.hx = torch.zeros(1, 256)
    
    def detach_hidden(self):
        self.cx = self.cx.detach()
        self.hx = self.hx.detach()

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
