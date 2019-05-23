import numpy as np
import torch
import torch.nn as nn
from collections import deque

import torch.nn.functional as F
from auxiliary_tasks.agent_wrapper import AgentWrapper

class ValueReplayWrapper(AgentWrapper):
    def __init__(self, agent, gamma=0.99):
        super(ValueReplayWrapper, self).__init__(agent)
        self._gamma = gamma

    def get_loss(self):
        return self.vr_loss() + self._agent.get_loss()

    def vr_loss(self):
        vr_experience_frames = self.sample_sequence(21)
        vr_experience_frames.reverse()

        batch_vr_si = []
        batch_vr_R = []
        batch_vr_last_action_reward = []

        vr_R = 0.0
        if not vr_experience_frames[1].terminal:
            with torch.no_grad():
                vr_R = self.vr_predict(
                    vr_experience_frames[0].state[None, :],
                    vr_experience_frames[0].get_last_action_reward(self.n_actions)[None, :]
                    ).detach().numpy()

        for frame in vr_experience_frames[1:]:
            vr_R = frame.reward + self._gamma * vr_R
            batch_vr_si.append(frame.state)
            batch_vr_R.append(vr_R)
            last_action_reward = frame.get_last_action_reward(self.n_actions)
            batch_vr_last_action_reward.append(last_action_reward)

        batch_vr_si.reverse()
        batch_vr_R.reverse()
        batch_vr_last_action_reward.reverse()

        batch_vr_R = torch.FloatTensor(batch_vr_R)
        batch_vr_si = np.array(batch_vr_si)
        batch_vr_last_action_reward = np.array(batch_vr_last_action_reward)
        vr_v = self.vr_predict(batch_vr_si, batch_vr_last_action_reward)

        return torch.sum((batch_vr_R - vr_v) ** 2 / 2.)

    def vr_predict(self, state, act_reward):
        state = torch.FloatTensor(state)
        act_reward = torch.FloatTensor(act_reward)
        cnn_out = self.cnn_features(state)
        x = cnn_out.view(-1, self.cnn_output_size)
        x = self.dense(x)
        x = F.relu(x)
        x = torch.cat((x, act_reward), 1)

        x, _ = self.lstm(x, None)
        return self.critic_linear(x)
