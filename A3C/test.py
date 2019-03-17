import logging
import numpy as np
import time

from openai_wrappers import make_atari
from model import ActorCritic
from utils import play_game


def save_progress(args, model, optimizer, steps):
    model_name = "{}_{}_{}".format(
        args.env_name,
        time.strftime("%Y.%m.%d_%H:%M", time.localtime()),
        steps
    )
    weights_path = "{}/weights/{}".format(args.models_path, model_name)
    model.save_weights(weights_path)
    log_message = "Wights were saved to {}".format(weights_path)
    print(log_message)
    logging.info(log_message)
                
    if optimizer is not None:
        optimizer_params_path = "{}/optimizer_params/{}".format(args.models_path, model_name)
        optimizer.save_params(optimizer_params_path)
        log_message = "Optimizer's params were saved to {}".format(optimizer_params_path)
        print(log_message)
        logging.info(log_message)


def test(args, shared_model, total_steps, optimizer):
    logging.basicConfig(filename=args.logs_path, level=logging.INFO)
    logging.info("STARTED TRAINING PROCESS {}".format(time.strftime("%Y.%m.%d_%H:%M", time.localtime())))

    env = make_atari(args.env_name, clip=False)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.eval()

    start_time = time.time()
    
    reward_history = []
    while True:
        model.load_state_dict(shared_model.state_dict())
        if (len(reward_history) + 1) % args.save_frequency == 0:
            save_progress(args, model, optimizer, total_steps.value)
        total_reward, _ = play_game(model, env)
        reward_history.append(total_reward)
        
        log_message = "Time {}, num steps {}, FPS {:.0f}, curr episode reward {}, mean episode reward: {}".format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
            total_steps.value,
            total_steps.value / (time.time() - start_time),
            total_reward,
            np.mean(reward_history[-60:]),
        )
        print(log_message)
        logging.info(log_message)
        time.sleep(60)

