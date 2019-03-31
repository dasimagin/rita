import argparse
import logging

from config import Config
from envs.utils import make_env
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from utils import record_video

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--config-path', required=True,
                    help='path to config')
parser.add_argument('--logs-path', default="./log.txt",
                    help='path to logs (default: `./log.txt`)')
parser.add_argument('--pretrained-weights', default=None,
                    help='path to pretrained weights (default: None – evaluate random model)')

if __name__ == '__main__':
    cmd_args = parser.parse_args()
    config = Config.fromYamlFile(cmd_args.config_path)
    
    logging.basicConfig(filename=cmd_args.logs_path, level=logging.INFO)

    env = make_env(config)
    model = ActorCritic(env.observation_space.shape, env.action_space.n)
    if cmd_args.pretrained_weights is not None:
        model.load_weights(cmd_args.pretrained_weights)
    else:
        print("You have not specified path to model weigths, random plays will be performed")
    model.eval()
    results = record_video(model, env)
    log_message = "evaluated on pretrained weights: {}, results: {}".format(cmd_args.pretrained_weights, results)
    print(log_message)
    logging.info(log_message)
