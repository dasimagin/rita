import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from auxiliary_tasks.agent_wrapper import AgentWrapper

def pc_reward(prev, next, size=20):
    f = prev.shape[1] // size
    f_rem = prev.shape[1] % size
    s = prev.shape[2] // size
    s_rem = prev.shape[2] % size
    f_rem = -prev.shape[1] if f_rem == 0 else f_rem
    s_rem = -prev.shape[2] if s_rem == 0 else s_rem
    prev = prev[:,:-f_rem,:-s_rem]
    next = next[:,:-f_rem,:-s_rem]
    a = np.mean(np.absolute(prev - next), 0)
    sh = a.shape
    sh = sh[0] // f, f, sh[1] // s, s
    return a.reshape(sh).mean(-1).mean(1)

class PixelControlWrapper(AgentWrapper):
    def __init__(self, agent, gamma=0.99, coef=0.001):
        super(PixelControlWrapper, self).__init__(agent)

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

    def get_loss(self):
        return self.pc_loss() + self._agent.get_loss()

    def pc_loss(self):
        gamma = self._gamma
        pc_experience_frames = self.sample_sequence(21)
        pc_experience_frames.reverse()
        batch_pc_si = []
        batch_pc_a = []
        batch_pc_R = []
        batch_pc_last_action_reward = []

        pc_R = np.zeros([20, 20], dtype=np.float32)
        if not pc_experience_frames[1].terminal:
            pc_R = self.pc_predict(pc_experience_frames[0].state[None, :],
                    pc_experience_frames[0].get_last_action_reward(self.n_actions)[None, :]
                    ).detach()
            pc_R = torch.max(pc_R, dim=1, keepdim=False)[0].numpy()

        for frame in pc_experience_frames[1:]:
            pc_R = frame.pixel_change + gamma * pc_R
            a = np.zeros([self.n_actions], dtype=np.float32)
            a[frame.action] = 1.0
            last_action_reward = frame.get_last_action_reward(self.n_actions)

            batch_pc_si.append(frame.state)
            batch_pc_a.append(a)
            batch_pc_R.append(pc_R)
            batch_pc_last_action_reward.append(last_action_reward)

        batch_pc_si.reverse()
        batch_pc_a.reverse()
        batch_pc_R.reverse()
        batch_pc_last_action_reward.reverse()

        batch_pc_si = np.array(batch_pc_si)
        batch_pc_a = torch.FloatTensor(np.array(batch_pc_a))
        batch_pc_R = torch.FloatTensor(np.array(batch_pc_R))
        batch_pc_last_action_reward = np.array(batch_pc_last_action_reward)
        pc_q = self.pc_predict(batch_pc_si, batch_pc_last_action_reward)

        pc_a_reshaped = batch_pc_a.view(-1, self.n_actions, 1, 1)
        pc_q = torch.mul(pc_q, pc_a_reshaped)
        pc_q = torch.sum(pc_q, dim=1, keepdim=False)
        return torch.mean(torch.sum((batch_pc_R - pc_q) ** 2 / 2., (1,2) )) * self._coef

    def pc_predict(self, state, act_reward):
        state = torch.FloatTensor(state)
        act_reward = torch.FloatTensor(act_reward)
        cnn_out = self.cnn_features(state)
        x = cnn_out.view(-1, self.cnn_output_size)
        x = self.dense(x)
        x = F.relu(x)
        x = torch.cat((x, act_reward), 1)

        x, _ = self.lstm(x, None)

        x = self.pc_map(x)
        x = x.view(-1, 32, 7, 7)
        v = F.relu(self.pc_deconv_V(x))
        a = F.relu(self.pc_deconv_A(x))
        a_mean = torch.mean(a, dim=1, keepdim=True)
        a = a-a_mean
        return v + a
