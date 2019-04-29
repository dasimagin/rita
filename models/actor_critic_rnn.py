import torch
import torch.nn as nn

from torch.nn.functional import softmax


def xavier_weights_init(layer):
    if type(layer) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_uniform_(layer.weight)


class ActorCriticRNN(torch.nn.Module):
    def __init__(self, observation_shape, n_actions):
        super(ActorCriticRNN, self).__init__()
        self.cnn_features = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU()
        )
        self.cnn_output_size = self._cnn_features_size(observation_shape)

        self.lstm_slow = nn.LSTMCell(256 * 2, 256)
        self.lstm_fast = nn.LSTMCell(self.cnn_output_size + n_actions + 1 + 256 * 2, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, n_actions)

        self.apply(xavier_weights_init)
        self.reset_hidden()

    def forward(self, observation, prev_action, prev_reward):
        x = self.cnn_features(observation)
        x = x.view(-1, self.cnn_output_size)

        if self.t % 10 == 0:
            self.hx_slow, self.cx_slow = self.lstm_slow(
                torch.cat([self.hx_fast, self.cx_fast], dim=1),
                (self.hx_slow, self.cx_slow)
            )
        self.hx_fast, self.cx_fast = self.lstm_fast(
            torch.cat([x, prev_action, prev_reward, self.hx_slow, self.cx_slow], dim=1), 
            (self.hx_fast, self.cx_fast)
        )
        value = self.critic_linear(self.hx_fast)
        logits = self.actor_linear(self.hx_fast)
        self.t += 1
        return value, logits

    def reset_hidden(self):
        self.t = 0
        self.cx_slow = torch.zeros(1, 256)
        self.hx_slow = torch.zeros(1, 256)
        self.cx_fast = torch.zeros(1, 256)
        self.hx_fast = torch.zeros(1, 256)

    def detach_hidden(self):
        self.cx_slow = self.cx_slow.detach()
        self.hx_slow = self.hx_slow.detach()
        self.cx_fast = self.cx_fast.detach()
        self.hx_fast = self.hx_fast.detach()

    def _cnn_features_size(self, input_shape):
        return self.cnn_features(torch.zeros(1, *input_shape)).view(1, -1).size(1)
