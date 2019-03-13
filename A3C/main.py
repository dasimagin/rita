import argparse

import torch
import torch.multiprocessing as mp

from openai_wrappers import make_atari
from model import ActorCritic
from shared_optim import SharedAdam
from test import test
from train import train

from multiprocessing import Value

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='maximum of gradient norm (default: 50)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--env-name', default='SpaceInvaders-v0',
                    help='train environment')
parser.add_argument('--save-frequency', type=int, default=60,
                    help='save model each `save_frequency` test steps (default: 60)')
parser.add_argument('--models-path', default="./saved_models",
                    help='path to saved models (default: `./saved_models`)')
parser.add_argument('--logs-path', default="./log.txt",
                    help='path to logs (default: `./log.txt`)')
parser.add_argument('--pretrained-weights', default=None,
                    help='path to pretrained weights and optimizer params (default: if None – train from scratch)')


if __name__ == '__main__':
    args = parser.parse_args()
    env = make_atari(args.env_name)
    
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    if args.pretrained_weights is not None:
        shared_model.load_weights(args.pretrained_weights)
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    if args.pretrained_weights is not None:
        optimizer.load_params(args.pretrained_weights.replace('weights/', 'optimizer_params/'))
    optimizer.share_memory()
        
    processes = []

    lock = mp.Lock()
    total_steps = Value('i', 0)

    p = mp.Process(target=test, args=(args, shared_model, total_steps, optimizer))
    p.start()
    processes.append(p)

    for _ in range(0, args.num_processes):
        p = mp.Process(target=train, args=(args, shared_model, total_steps, optimizer, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
