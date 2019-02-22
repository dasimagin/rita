import torch
import torch.optim as optim

import gym
import math, random
import numpy as np

class DoubleDQAgent:
    def __init__(self, env, net, target_net, replay_buffer, epsilon=0):
        self.env = env
        self.net = net
        self.target_net = target_net
        self.epsilon = epsilon
        self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=0.0002)
        self.replay_buffer = replay_buffer
        self.epsilon = epsilon
        self.Variable = lambda *args, **kwargs: torch.autograd.Variable(*args, **kwargs).cuda() if torch.cuda.is_available() else autograd.Variable(*args, **kwargs) 
    
    def synchonize_target(self):
        self.target_net.load_state_dict(self.net.state_dict())
        
    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        with torch.no_grad():
            if random.random() > epsilon:
                state   = self.Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
                q_value = self.net(state)
                action  = q_value.max(1)[1].item()
            else:
                action = random.randrange(self.env.action_space.n)
        return action
    
    def train(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = self.Variable(torch.FloatTensor(np.float32(state)))
        next_state = self.Variable(torch.FloatTensor(np.float32(next_state)))
        action     = self.Variable(torch.LongTensor(action))
        reward     = self.Variable(torch.FloatTensor(reward))
        done       = self.Variable(torch.FloatTensor(done))

        q_values      = self.net(state)
        next_q_values = self.net(next_state)
        next_q_state_values = self.target_net(next_state) 

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + GAMMA * next_q_value * (1 - done)

        loss = (q_value - self.Variable(expected_q_value.data)).pow(2).mean()
        return loss, q_values.mean()
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def play_game(self, env=None):
        if env is None:
            env = self.env
        state = env.reset()
        is_done = False
        total_reward = 0.0
        
        self.epsilon = 0.05
        while not is_done:
            action = self.get_action(state)
            state, reward, is_done, _ = env.step(action)
            total_reward += reward
        
        return total_reward
    
    def push_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def record_video(self, dir_path='videos'):
        env_monitor = gym.wrappers.Monitor(self.env, directory=dir_path, force=True)
        session = self.play_game(env_monitor)
        env_monitor.close()
    
    def save_net(self, path='ddqn_weights'):
        torch.save(self.net.state_dict(), path)
    
    def load_net(self, path='ddqn_weights'):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        
        self.target_net.load_state_dict(torch.load(path))
        self.target_net.eval()
        
        
class AsyncDDQAgent(DoubleDQAgent):
    def __init__(self, net, target, env):
        self.env = env
        self.net = net
        self.target_net = target
        
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=0.0002)
        self.epsilon = 0
        self.Variable = lambda *args, **kwargs: torch.autograd.Variable(*args, **kwargs).cuda() if torch.cuda.is_available() else autograd.Variable(*args, **kwargs) 

    def max_qvalue(self, s, net_type='main'):
        s = self.Variable(torch.FloatTensor(np.float32(s)).unsqueeze(0))
        if net_type == 'main':
            qvalues = self.net(s)
        else:
            with torch.no_grad():
                qvalues = self.target_net(s)
        return qvalues.max(1)[0]
        
    def qvalue(self, s, a):
        s = self.Variable(torch.FloatTensor(np.float32(s)).unsqueeze(0))
        qvalues = self.net(s)
        return qvalues[0][a]
        
        