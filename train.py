import argparse
import torch
import torch.multiprocessing as mp

from config import Config
from envs.utils import make_env
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from auxiliary_tasks.pixel_control import PixelControlWrapper
from auxiliary_tasks.agent_wrapper import BaseWrapper
from auxiliary_tasks.reward_prediction import RewardPredictionWrapper
from auxiliary_tasks.value_replay import ValueReplayWrapper
from auxiliary_tasks.experience import *
from optim.shared_optim import SharedAdam
from workers import test_worker, train_worker

from multiprocessing import Value

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--experiment-folder', required=True,
                    help='path to folder with config (here weights and log will be stored)')
parser.add_argument('--pretrained-weights', default=None,
                    help='path to pretrained weights (default: if None – train from scratch)')
parser.add_argument('--pretrained-optimizer', default=None,
                    help='path to pretrained optimizer params (default: if None – train from scratch)')

if __name__ == '__main__':
    cmd_args = parser.parse_args()
    config = Config.fromYamlFile('{}/{}'.format(cmd_args.experiment_folder, 'config.yaml'))
    config.train.__dict__.update(vars(cmd_args))

    env = make_env(config.environment)

    shared_model = ActorCritic(env.observation_space.shape, env.action_space.n)
    shared_model = BaseWrapper(shared_model)
    if (config.train.use_pixel_control or
            config.train.use_reward_prediction or
            config.train.use_value_replay):
        shared_model = ExperienceWrapper(shared_model)
    if config.train.use_pixel_control:
        shared_model = PixelControlWrapper(shared_model, config.train.gamma, config.train.pc_coef)
    if config.train.use_reward_prediction:
        shared_model = RewardPredictionWrapper(shared_model, config.train.rp_coef)
    if config.train.use_value_replay:
        shared_model = ValueReplayWrapper(shared_model)
    if config.train.pretrained_weights is not None:
        shared_model.load_state_dict(torch.load(config.train.pretrained_weights))
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=config.train.learning_rate)
    if config.train.pretrained_optimizer is not None:
        optimizer.load_state_dict(torch.load(config.train.pretrained_optimizer))
    optimizer.share_memory()

    processes = []

    lock = mp.Lock()
    total_steps = Value('i', 0)

    p = mp.Process(target=test_worker, args=(config, shared_model, total_steps, optimizer))
    p.start()
    processes.append(p)

    for _ in range(0, config.train.agents_n):
        p = mp.Process(target=train_worker, args=(config, shared_model, total_steps, optimizer, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
