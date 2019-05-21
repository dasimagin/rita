import numpy as np
import torch
import torch.nn as nn
from collections import deque

import torch.nn.functional as F
from auxiliary_tasks.agent_wrapper import AgentWrapper

class RewardBuffer:
    def __init__(self, size=2000):
        self.size = size
        self.no_reward = deque(maxlen=size)
        self.with_reward = deque(maxlen=size)
        self.target = deque(maxlen=size)

    def add(self, obs, sign=0):
        if sign == 0:
            self.no_reward.append([o.detach() for o in obs])
        else:
            self.with_reward.append([o.detach() for o in obs])
            self.target.append(1 if sign > 0 else 2)

    def sample(self):
        if (len(self.no_reward)+len(self.with_reward) >= self.size and
                len(self.no_reward) > 0 and
                len(self.with_reward) > 0):

            if np.random.choice([0, 1]) == 0:
                ind = np.random.choice(len(self.no_reward))
                return self.no_reward[ind], 0
            else:
                ind = np.random.choice(len(self.with_reward))
                return self.with_reward[ind], self.target[ind]
        else:
            return None, None


class RewardPredictionWrapper(AgentWrapper):
    def __init__(self, agent):
        super(RewardPredictionWrapper, self).__init__(agent)

        self._last_frames = [torch.zeros(self.ob_shape).unsqueeze(0)
                             for i in range(4)]
        self._last_reward = 0
        self._buffer = RewardBuffer()

        self.rp_linear = nn.Linear(self.cnn_output_size*4, 128)
        self.rp_pred = nn.Linear(128, 3)

    def forward(self, x):
        ret = self._agent(x)

        self._buffer.add(self._last_frames, x[2]-self._last_reward)
        self._last_reward = x[2]
        self._last_frames.pop(0)
        self._last_frames.append(x[0])

        return ret

    def get_loss(self):
        return self.rp_loss() + self._agent.get_loss()

    def reset(self):
        self._last_frames = [torch.zeros(self.ob_shape).unsqueeze(0)
                             for i in range(4)]
        self._last_reward = 0
        self._agent.reset()

    def rp_loss(self):
        obs, ind = self._buffer.sample()
        if obs is None:
            return torch.tensor(0)

        pred = self.rp_predict(obs)
        target = torch.zeros([1, 3])
        target[0][ind] = 1
        loss = -torch.sum(torch.log(pred)*target)
        return loss

    def rp_predict(self, obs):
        en1 = self.cnn_features(obs[0]).view(-1, self.cnn_output_size)
        en2 = self.cnn_features(obs[1]).view(-1, self.cnn_output_size)
        en3 = self.cnn_features(obs[2]).view(-1, self.cnn_output_size)
        en4 = self.cnn_features(obs[3]).view(-1, self.cnn_output_size)
        obs = torch.cat((en1, en2, en3, en4), 1)

        pred = self.rp_linear(obs)
        pred = F.relu(pred)
        pred = self.rp_pred(pred)
        pred = F.softmax(pred, dim=1)

        return pred
