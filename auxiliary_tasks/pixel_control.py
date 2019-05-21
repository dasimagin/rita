import torch
import torch.nn as nn

import torch.nn.functional as F
from auxiliary_tasks.agent_wrapper import AgentWrapper

def pc_reward(prev, next, size=20):
    x = torch.empty([prev.shape[0], size, size])
    h = int(prev.shape[2]/size)
    w = int(prev.shape[3]/size)
    for i in range(size):
        for j in range(size):
            prev_part = prev[:,:,i*h:(i+1)*h, j*w:(j+1)*w]
            next_part = next[:,:,i*h:(i+1)*h, j*w:(j+1)*w]
            abs_diff = torch.abs(prev_part-next_part)
            x[:,i,j] = torch.mean(abs_diff, [-1, -2, -3])
    return x.detach()

class PixelControlWrapper(AgentWrapper):
    def __init__(self, agent, gamma=0.99, coef=0.001):
        super(PixelControlWrapper, self).__init__(agent)

        self._cur_rewards = []
        self._prev_state = None
        self._first_act = None
        self._first_state_encoded = None
        self._last_state_encoded = None
        self._gamma = gamma
        self._coef = coef

        self.pc_map = nn.Linear(256, 32*7*7)
        self.pc_deconv_V = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=4,
            stride=3,
            padding=1)
        self.pc_deconv_A = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=self.n_actions,
            kernel_size=4,
            stride=3,
            padding=1)

    def forward(self, x):
        ret = self._agent(x)

        if self._prev_state is not None:
            self._cur_rewards.append(pc_reward(self._prev_state, x[0]))
        else:
            self._first_act = x[1]
            self._first_state_encoded = self.hx
        self._prev_state = x[0]
        self._last_state_encoded = self.hx
        return ret

    def get_loss(self):
        return self.pc_loss() + self._agent.get_loss()

    def reset(self):
        self._cur_rewards = []
        self._prev_state = None
        self._first_state_encoded = None
        self._last_state_encoded = None
        self._agent.reset()

    def pc_loss(self):
        gamma = self._gamma
        sum_reward = 0
        last_coef = 1
        first_pred = self.pc_predict(self._first_state_encoded)
        last_pred = self.pc_predict(self._last_state_encoded)
        for reward in reversed(self._cur_rewards):
            sum_reward *= gamma
            sum_reward += reward
            last_coef *= gamma
        last_pred = torch.max(last_pred, 1)[0] * last_coef
        sum_reward += last_pred
        first_pred = first_pred[:, self._first_act]
        return torch.sum(torch.pow(first_pred - sum_reward.detach(), 2)/2) * self._coef

    def pc_predict(self, x):
        x = self.pc_map(x)
        x = F.relu(x)
        x = x.view(-1, 32, 7, 7)
        v = F.relu(self.pc_deconv_V(x))
        a = F.relu(self.pc_deconv_A(x))
        a_mean = torch.mean(a, dim=1, keepdim=True)
        a = a-a_mean
        return v + a
