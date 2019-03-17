import gym
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
