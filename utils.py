import gym.wrappers
import logging
import time
import torch
import torch.nn.functional as F
import numpy as np
import os


def play_game(model, env):
    args = model.config.train
    state = env.reset()
    model.reset_hidden()
    done = False
    total_reward = 0.0
    episode_length = 0

    values = []
    log_probs = []
    entropies = []
    rewards = []
    last_act = 0
    sum_reward = 0

    while not done:
        state = torch.FloatTensor(state)
        with torch.no_grad():
            value, logit = model.forward((state.unsqueeze(0), last_act, sum_reward))
        prob = F.relu(F.softmax(logit, dim=-1))
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        action = prob.multinomial(num_samples=1).detach()
        log_prob = log_prob.gather(1, action)

        state, reward, done, _ = env.step(prob[0].max(0)[1].item())
        last_act = prob[0].max(0)[1].item()
        sum_reward += reward

        entropies.append(entropy.numpy()[0][0])
        values.append(value.numpy()[0][0])
        log_probs.append(log_prob.numpy()[0][0])
        rewards.append(reward)
        total_reward += reward

    env.reset()
    R = 0
    values.append(R)
    policy_loss = 0
    value_loss = 0
    entropy = 0
    gae = 0
    ep_len = len(rewards)
    for i in reversed(range(ep_len)):
        R = args.gamma * R + rewards[i]
        advantage = R - values[i]
        value_loss = value_loss + 0.5 * (advantage**2)

        # Generalized Advantage Estimataion
        delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
        gae = gae * args.gamma * args.tau + delta_t

        policy_loss -= log_probs[i] * gae
        entropy += entropies[i]

    stats = dict()
    stats['total_reward'] = total_reward
    stats['ep_len'] = ep_len
    stats['policy_loss'] = policy_loss / ep_len
    stats['value_loss'] = value_loss / ep_len
    stats['entropy'] = entropy / ep_len
    max_entropy = np.log(env.action_space.n)
    stats['entropy'] /= max_entropy
    if model.config.train.use_pixel_control:
        stats['pc_loss'] = model.pc_loss().detach().numpy()
    if model.config.train.use_reward_prediction:
        stats['rp_loss'] = model.rp_loss().detach().numpy()

    model.reset()

    return stats


def record_video(model, env):
    env_monitor = gym.wrappers.Monitor(env, directory='videos', force=True)
    reward, length, policy_loss, value_loss, entropy = play_game(model, env_monitor)
    stats = play_game(model, env_monitor)
    env_monitor.close()
    result = {
        'reward': stats['total_reward'],
        'len': stats['ep_len'],
        'mean policy loss': stats['policy_loss'],
        'mean value loss': stats['value_loss'],
        'mean entropy percentage': stats['entropy']
    }
    return result


def save_progress(args, model, optimizer, steps):
    model_name = "{}_{}_{}".format(
        args.environment.env_name,
        time.strftime("%Y.%m.%d_%H:%M", time.localtime()),
        steps
    )
    if not os.path.exists("{}/weights/".format(args.train.experiment_folder)):
        os.makedirs("{}/weights/".format(args.train.experiment_folder))
    if not os.path.exists("{}/optimizer_params/".format(args.train.experiment_folder)):
        os.makedirs("{}/optimizer_params/".format(args.train.experiment_folder))
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
