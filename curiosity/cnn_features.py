import torch
import torch.nn as nn

class SimpleConvNet(nn.Module):
    def __init__(self, observation_shape):
        super(SimpleConvNet, self).__init__()
        self.cnn_features = nn.Sequential(
            nn.Conv2d(observation_shape[0], 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.cnn_output_size = self._cnn_features_size(observation_shape)
        
    def forward(self, x):
        x = self.cnn_features(x)
        x = x.view(-1, self.cnn_output_size)
        return x
    
    def _cnn_features_size(self, input_shape):
        return self.cnn_features(torch.zeros(1, *input_shape)).view(1, -1).size(1)


