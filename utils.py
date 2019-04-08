import gym.wrappers
import logging
import time
import torch
import torch.nn.functional as F


def play_game(model, env):
    args = model.config
    state = env.reset()
    model.reset_hidden()
    done = False
    total_reward = 0.0
    episode_length = 0

    values = []
    log_probs = []
    entropies = []
    rewards = []

    while not done:
        state = torch.FloatTensor(state)
        with torch.no_grad():
            value, logit = model.forward(state.unsqueeze(0))
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        action = prob.multinomial(num_samples=1).detach()
        log_prob = log_prob.gather(1, action)

        state, reward, done, _ = env.step(action[0, 0])

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

    policy_loss /= ep_len
    value_loss /= ep_len
    entropy /= ep_len
    max_entropy = np.log(env.action_space.n)
    entropy /= max_entropy

    return total_reward, ep_len, policy_loss, value_loss, entropy


def record_video(model, env, games_count=2):
    env_monitor = gym.wrappers.Monitor(env, directory='videos', force=True)
    results = []
    for _ in range(games_count):
        reward, length, policy_loss, value_loss, entropy = play_game(model, env_monitor)
        results.append({
            'reward': reward,
            'len': length,
            'mean policy loss': policy_loss,
            'mean value loss': value_loss,
            'mean entropy percentage': entropy
        })
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
