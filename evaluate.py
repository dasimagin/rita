import argparse
import logging

from envs.common_wrappers import make_atari
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from utils import record_video

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--env-name', default='SpaceInvaders-v0',
                    help='environment to train on (default: SpaceInvaders-v0)')
parser.add_argument('--logs-path', default="./log.txt",
                    help='path to logs (default: `./log.txt`)')
parser.add_argument('--pretrained-weights', default=None,
                    help='path to pretrained weights (default: None – evaluate random model)')

if __name__ == '__main__':
    cmd_args = parser.parse_args()
    logging.basicConfig(filename=cmd_args.logs_path, level=logging.INFO)

    env = make_atari(cmd_args.env_name, clip=False)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    if cmd_args.pretrained_weights is not None:
        model.load_weights(cmd_args.pretrained_weights)
    else:
        print("You have not specified path to model weigths, random plays will be performed")
    model.eval()
    results = record_video(model, env)
    log_message = "evaluated on pretrained weights: {}, results: {}".format(cmd_args.pretrained_weights, results)
    print(log_message)
    logging.info(log_message)
