import argparse
import torch.multiprocessing as mp

from config import Config
from envs.common_wrappers import make_atari
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from optim.shared_optim import SharedAdam
from workers import test_worker, train_worker

from multiprocessing import Value

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--config-path', required=True,
                    help='path to config')
parser.add_argument('--models-path', default="./saved_models",
                    help='path to saved models (default: `./saved_models`)')
parser.add_argument('--logs-path', default="./log.txt",
                    help='path to logs (default: `./log.txt`)')
parser.add_argument('--pretrained-weights', default=None,
                    help='path to pretrained weights and optimizer params (default: if None – train from scratch)')

if __name__ == '__main__':
    cmd_args = parser.parse_args()
    config = Config.fromYamlFile(cmd_args.config_path)
    args.train.__dict__.update(vars(cmd_args))

    env = make_atari(args.env_name)

    shared_model = ActorCritic(env.observation_space.shape, env.action_space.n)
    if args.pretrained_weights is not None:
        shared_model.load_weights(args.pretrained_weights)
    shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=args.learning_rate)
    if args.pretrained_weights is not None:
        optimizer.load_params(args.pretrained_weights.replace('weights/', 'optimizer_params/'))
    optimizer.share_memory()

    processes = []

    lock = mp.Lock()
    total_steps = Value('i', 0)

    p = mp.Process(target=test_worker, args=(args, shared_model, total_steps, optimizer))
    p.start()
    processes.append(p)

    for _ in range(0, args.agents_n):
        p = mp.Process(target=train_worker, args=(args, shared_model, total_steps, optimizer, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
