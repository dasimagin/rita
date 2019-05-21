import torch.nn as nn

class AgentWrapper(nn.Module):
    def __init__(self, agent):
        super(AgentWrapper, self).__init__()
        self._agent = agent

    def forward(self, x):
        return self._agent(x)

    def __getattr__(self, name):
        if name == '_agent':
            return self.__dict__['_modules']['_agent']
        if name in self.__dict__['_modules']:
            return self.__dict__['_modules'][name]
        return getattr(self.__dict__['_modules']['_agent'], name)

    def __call__(self, x):
        return self.forward(x)

class BaseWrapper(AgentWrapper):
    def __init__(self, agent):
        super(BaseWrapper, self).__init__(agent)

    def get_loss(self):
        return 0

    def reset(self):
        return
