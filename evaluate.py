import argparse
import logging
import torch

from config import Config
from envs.utils import make_env
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from utils import record_video

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--experiment-folder', required=True,
                    help='path to folder with config')
parser.add_argument('--pretrained-weights', default=None,
                    help='path to pretrained weights (default: None – evaluate random model)')

if __name__ == '__main__':
    cmd_args = parser.parse_args()
    config = Config.fromYamlFile('{}/{}'.format(cmd_args.experiment_folder, 'config.yaml'))

    log_path = '{}/{}'.format(cmd_args.experiment_folder, 'log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO)

    config.environment.clip_rewards = False
    env = make_env(config.environment)
    model = ActorCritic(env.observation_space.shape, env.action_space.n)
    model.config = config
    if cmd_args.pretrained_weights is not None:
        model.load_state_dict(torch.load(cmd_args.pretrained_weights))
    else:
        print("You have not specified path to model weigths, random plays will be performed")
    model.eval()
    results = record_video(model, env)
    log_message = "evaluated on pretrained weights: {}, results: {}".format(cmd_args.pretrained_weights, results)
    print(log_message)
    logging.info(log_message)
