import argparse
import torch
import torch.multiprocessing as mp

from config import Config
from envs.utils import make_env
from genetic_alorithms import GeneticOptimizer
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from optim.shared_optim import SharedAdam
from workers import test_worker, train_worker

from multiprocessing import Value
from multiprocessing.managers import BaseManager

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--experiment-folder', required=True,
                    help='path to folder with config (here weights and log will be stored)')
parser.add_argument('--pretrained-weights', default=None,
                    help='path to pretrained weights (default: if None – train from scratch)')
parser.add_argument('--pretrained-optimizer', default=None,
                    help='path to pretrained optimizer params (default: if None – train from scratch)')

class GeneticOptimizerManager(BaseManager):  
    pass  

GeneticOptimizerManager.register('GeneticOptimizer', GeneticOptimizer, exposed = ['push', 'pull']) 

if __name__ == '__main__':
    cmd_args = parser.parse_args()
    config = Config.fromYamlFile('{}/{}'.format(cmd_args.experiment_folder, 'config.yaml'))
    config.train.__dict__.update(vars(cmd_args))

    env = make_env(config.environment)

    shared_model = ActorCritic(env.observation_space.shape, env.action_space.n)
    if config.train.pretrained_weights is not None:
        shared_model.load_state_dict(torch.load(config.train.pretrained_weights))
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=config.train.learning_rate)
    if config.train.pretrained_optimizer is not None:
        optimizer.load_state_dict(torch.load(config.train.pretrained_optimizer))
    optimizer.share_memory()

    genetic_optimzer_manager = GeneticOptimizerManager()  
    genetic_optimzer_manager.start()  
    genetic_optimizer = genetic_optimzer_manager.GeneticOptimizer()
    
    processes = []

    lock = mp.Lock()
    total_steps = Value('i', 0)

    p = mp.Process(target=test_worker, args=(config, shared_model, total_steps, optimizer))
    p.start()
    processes.append(p)

    for _ in range(0, config.train.agents_n):
        p = mp.Process(target=train_worker, args=(config, shared_model, total_steps, optimizer, lock, genetic_optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
