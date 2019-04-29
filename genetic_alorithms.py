from config import Config

import numpy as np
import torch.multiprocessing as mp

HYPERPARAM_NAMES = ['curiosity_weight', 'entropy_weight', 'gamma', 'learning_rate', 'tau', 'value_weight']

def crossover(a, b):
    child = Hyperparams({})
    for param in a.__dict__.keys():
        if np.random.uniform() < 0.5:
            child.__dict__[param] = a.__dict__[param]
        else:
            child.__dict__[param] = b.__dict__[param]
    return child


def mutate(a):
    mutated = Hyperparams({})
    for param, value in a.__dict__.items():
        if np.random.uniform() < 0.05:
            mutated.__dict__[param] = min(value * 10, 0.999)
        elif np.random.uniform() < 0.05:
            mutated.__dict__[param] = value / 10
        elif np.random.uniform() < 0.3:
            mutated.__dict__[param] = np.random.uniform(value * 0.9, min(value * 1.1, 1))
        else:
            mutated.__dict__[param] = value
    return mutated


class Hyperparams(Config):
    def __init__(self, config):
        dict_config = {}
        if isinstance(config, Config):
            dict_config = config.__dict__
        elif isinstance(config, dict):
            dict_config = config
        else:
            raise NotImplemented
        hyperparams = {}
        for key, value in dict_config.items():
            if key in HYPERPARAM_NAMES:
                hyperparams[key] = value
        super(Hyperparams, self).__init__(hyperparams)
        
    def __add__(self, other):
        assert isinstance(other, Hyperparams)
        add_result = Hyperparams({})
        for param in self.__dict__.keys():
            add_result.__dict__[param] = self.__dict__[param] + other.__dict__[param]
        return add_result
    
    def __truediv__(self, number):
        assert isinstance(number, int)
        div_result = Hyperparams({})
        for param in self.__dict__.keys():
            div_result.__dict__[param] = self.__dict__[param] / number
        return div_result
    
    def __str__(self):
        return str(self.__dict__)

class GeneticOptimizer:
    def __init__(self):
        self.params = []
        self.results = []
        self.mutex = mp.Lock()
        
    def push(self, param, result):
        with self.mutex:
            self.params.append(param)
            self.results.append(result)
            if len(self.params) > 300:
                self.params = self.params[-250:]
                self.results = self.results[-250:]
        
    def pull(self):
        with self.mutex:
            if len(self.params) < 30:
                return self.params[0]
            best = np.argsort(self.results)[-30:]
            first_parent, second_parent = np.random.choice(best, size=2, replace=False)
            child = crossover(self.params[first_parent], self.params[second_parent])
            mutated_child = mutate(child)
            id_successfull = np.argsort(self.results)[-1]
            print('Most successfull', self.params[id_successfull], self.results[id_successfull])
            return mutated_child
