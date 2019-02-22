import os.path
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from multiprocessing import Value
from gym_utils import make_env
from agents import AsyncDDQAgent
from nets import ConvQNet

N_PROCESSES = 6

LEARNING_RATE = 0.0002

TARGET_UPDATE_FREQ = 40000
ASYNC_UPDATE_FREQ = 5
MAX_STEPS_TOTAL = 1500000
GAMMA = 0.99

def train(net, target_net, total_steps, epsilon):
    # Construct data_loader, optimizer, etc.
    env = make_env('SpaceInvaders-v0', clip=True)
    unclipped_env = make_env('SpaceInvaders-v0', clip=False)
    model = AsyncDDQAgent(net, target_net, env)
    
    t = 0
    s = env.reset()
    total_loss = 0
    sum_y = 0
    while total_steps.value < MAX_STEPS_TOTAL:
        a = model.get_action(s, epsilon)
        next_s, reward, done, _ = env.step(a)
        if done:
            y = reward
            s = env.reset()
        else:
            y = (reward + GAMMA * model.max_qvalue(next_s, net_type='target')).item()
            s = next_s
        sum_y += y
        total_loss += (y - model.qvalue(s, a))**2
        t += 1
        with total_steps.get_lock():
            total_steps.value += 1
        if total_steps.value % TARGET_UPDATE_FREQ == 0:
            model.synchonize_target()
        if (t + 1) % ASYNC_UPDATE_FREQ == 0:
            model.update(total_loss / ASYNC_UPDATE_FREQ)
            total_loss = 0
        if t % 5000 == 0:
            print('total_steps: {}, t: {}, loss: {}, epsilon: {}, mean_y: {}, game_reward: {}'.format(
                total_steps.value, t, total_loss, epsilon, sum_y / 5000, model.play_game()
            ))
            sum_y = 0
        
        
if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    
    path = 'async_qlearning_weights2'
    
    total_steps = Value('i', 0)
    
    env = make_env('SpaceInvaders-v0', clip=False)
    
    net = ConvQNet(env.observation_space.shape, env.action_space.n).to('cuda')
    net.share_memory()
    if os.path.isfile(path):
        net.load_state_dict(torch.load(path))
    net.eval()
    target_net = ConvQNet(env.observation_space.shape, env.action_space.n).to('cuda')
    if os.path.isfile(path):
        target_net.load_state_dict(torch.load(path))
    target_net.share_memory()

    processes = []
    for i in range(N_PROCESSES):
        p = mp.Process(target=train, args=(net, target_net, total_steps, (i + 1) * 0.08))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    torch.save(net.state_dict(), path)
        
