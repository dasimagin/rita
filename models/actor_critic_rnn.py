import torch
import torch.nn as nn

from torch.nn.functional import softmax
import torch.nn.functional as F

def xavier_weights_init(layer):
    if type(layer) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_uniform_(layer.weight)


class ActorCriticRNN(torch.nn.Module):
    def __init__(self, observation_shape, n_actions):
        super(ActorCriticRNN, self).__init__()
        self.ob_shape = observation_shape
        self.n_actions = n_actions

        self.cnn_features = nn.Sequential(
            nn.Conv2d(observation_shape[0], 16, 8, stride=4),
            nn.ELU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ELU()
        )
        self.cnn_output_size = self._cnn_features_size(observation_shape)

        self.dense = nn.Linear(self.cnn_output_size, 256)

        self.lstm = nn.LSTMCell(256 + n_actions + 1, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, n_actions)

        self.apply(xavier_weights_init)
        self.reset_hidden()

    def forward(self, inputs):
        act = torch.zeros(inputs[0].shape[0], self.n_actions)
        act[0, inputs[1]] = 1
        reward = torch.FloatTensor([[inputs[2]]])
        inputs = inputs[0]
        self.cnn_out = self.cnn_features(inputs)
        x = self.cnn_out.view(-1, self.cnn_output_size)
        x = self.dense(x)
        x = F.relu(x)
        x = torch.cat((x, act, reward), 1)

        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        value = self.critic_linear(self.hx)
        logits = self.actor_linear(self.hx)
        return value, logits

    def reset_hidden(self):
        self.cx = torch.zeros(1, 256)
        self.hx = torch.zeros(1, 256)

    def detach_hidden(self):
        self.cx = self.cx.detach()
        self.hx = self.hx.detach()

    def _cnn_features_size(self, input_shape):
        return self.cnn_features(torch.zeros(1, *input_shape)).view(1, -1).size(1)
