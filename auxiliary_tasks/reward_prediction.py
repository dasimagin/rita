import numpy as np
import torch
import torch.nn as nn
from collections import deque

import torch.nn.functional as F
from auxiliary_tasks.agent_wrapper import AgentWrapper


class RewardPredictionWrapper(AgentWrapper):
    def __init__(self, agent, coef=0.1):
        super(RewardPredictionWrapper, self).__init__(agent)

        self._coef = coef

        self.rp_linear = nn.Linear(self.cnn_output_size*3, 128)
        self.rp_pred = nn.Linear(128, 3)

    def get_loss(self):
        return self.rp_loss() + self._agent.get_loss()

    def rp_loss(self):
        rp_experience_frames = self.sample_rp_sequence()

        batch_rp_si = []
        batch_rp_c = []

        for i in range(3):
            batch_rp_si.append(rp_experience_frames[i].state)

        r = rp_experience_frames[3].reward
        rp_c = [0.0, 0.0, 0.0]
        if r == 0:
            rp_c[0] = 1.0
        elif r > 0:
            rp_c[1] = 1.0
        else:
            rp_c[2] = 1.0
        batch_rp_c.append(rp_c)
        batch_rp_si, batch_rp_c = np.array(batch_rp_si), np.array(batch_rp_c)

        rp_c = self.rp_predict(batch_rp_si)

        rp_c = torch.clamp(rp_c, 1e-20, 1.0)
        batch_rp_c = torch.FloatTensor(batch_rp_c)
        rp_loss = -torch.sum(batch_rp_c * torch.log(rp_c))
        return rp_loss * self._coef

    def rp_predict(self, obs):
        obs = torch.FloatTensor(obs)
        cnn_out = self.cnn_features(obs)
        cnn_out = cnn_out.view(1, -1)
        rp = self.rp_linear(cnn_out)
        rp = F.relu(rp)
        rp = self.rp_pred(rp)
        rp = F.softmax(rp, dim=1)
        return rp
