import gym.wrappers
import logging
import time
import torch
import torch.nn.functional as F


def play_game(model, env):
    state = env.reset()
    model.reset_hidden()
    done = False
    total_reward = 0.0
    episode_length = 0
    while not done:
        state = torch.FloatTensor(state)
        with torch.no_grad():
            value, logit = model.forward(state.unsqueeze(0))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()
        state, reward, done, _ = env.step(action[0, 0])
        total_reward += reward
        episode_length += 1
    return total_reward, episode_length


def record_video(model, env, games_count=2):
    env_monitor = gym.wrappers.Monitor(env, directory='videos', force=True)
    results = []
    for _ in range(games_count):
        reward, length = play_game(model, env_monitor)
        results.append({'reward': reward, 'len': length})
    env_monitor.close()
    return results


def save_progress(args, model, optimizer, steps):
    model_name = "{}_{}_{}".format(
        args.environment.env_name,
        time.strftime("%Y.%m.%d_%H:%M", time.localtime()),
        steps
    )
    weights_path = "{}/weights/{}".format(args.train.experiment_folder, model_name)
    torch.save(model.state_dict(), weights_path)
    log_message = "Weights were saved to {}".format(weights_path)
    print(log_message)
    logging.info(log_message)

    optimizer_params_path = "{}/optimizer_params/{}".format(args.train.experiment_folder, model_name)
    torch.save(optimizer.state_dict(), optimizer_params_path)
    log_message = "Optimizer's params were saved to {}".format(optimizer_params_path)
    print(log_message)
    logging.info(log_message)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            shared_param._grad += param.grad
        else:
            shared_param._grad = param.grad
