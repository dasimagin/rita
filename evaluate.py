import argparse
import logging
import torch

from config import Config
from envs.utils import make_env
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from utils import record_video

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--experiment-path', required=True,
                    help='path to folder with config')
parser.add_argument('--pretrained-weights', default=None,
                    help='path to pretrained weights (default: None – evaluate random model)')

if __name__ == '__main__':
    cmd_args = parser.parse_args()
    config_path = '{}/{}'.format(cmd_args.experiment_path, 'config.yaml')
    config = Config.fromYamlFile(config_path)
    
    log_path = '{}/{}'.format(cmd_args.experiment_path, 'log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO)

    env = make_env(config)
    model = ActorCritic(env.observation_space.shape, env.action_space.n)
    if cmd_args.pretrained_weights is not None:
        model.load_state_dict(torch.load(cmd_args.pretrained_weights))
    else:
        print("You have not specified path to model weights, random plays will be performed")
    model.eval()
    results = record_video(model, env)
    log_message = "evaluated on pretrained weights: {}, results: {}".format(cmd_args.pretrained_weights, results)
    print(log_message)
    logging.info(log_message)
