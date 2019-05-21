import logging
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from curiosity.model import CuriosityRewarder
from envs.utils import make_env
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from auxiliary_tasks.pixel_control import PixelControlWrapper
from auxiliary_tasks.agent_wrapper import BaseWrapper
from auxiliary_tasks.reward_prediction import RewardPredictionWrapper
from utils import ensure_shared_grads, play_game, save_progress


def train_worker(args, shared_model, total_steps, optimizer, lock):
    env = make_env(args.environment)
    args = args.train
    if args.sample_entropy:
        args.entropy_weight = np.exp(
            np.random.uniform(np.log(0.0005), np.log(0.01)))
    if args.sample_lr:
        args.learning_rate = np.exp(
            np.random.uniform(np.log(0.0001), np.log(0.005)))

    model = ActorCritic(env.observation_space.shape, env.action_space.n)
    model = BaseWrapper(model)
    if args.use_pixel_control:
        model = PixelControlWrapper(model, args.gamma, args.pc_coef)
    if args.use_reward_prediction:
        model = RewardPredictionWrapper(model)
    model.train()

    curiosity_rewarder = CuriosityRewarder(env.observation_space.shape, env.action_space.n)
    curiosity_rewarder.train()

    curiosity_optimizer = optim.Adam(curiosity_rewarder.parameters())

    state = env.reset()
    state = torch.FloatTensor(state)
    last_act = 0
    sum_reward = 0

    while True:
        model.load_state_dict(shared_model.state_dict())
        model.detach_hidden()

        values = []
        log_probs = []
        rewards = []
        curiosity_rewards = []
        entropies = []

        for step in range(args.update_agent_frequency):
            value, logit = model((state.unsqueeze(0), last_act, sum_reward))
            prob = F.relu(F.softmax(logit, dim=-1))
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            last_act = action.numpy()[0][0]
            next_state, reward, done, _ = env.step(last_act)
            sum_reward += reward

            with total_steps.get_lock():
                total_steps.value += 1

            if done:
                sum_reward = 0
                last_act = 0
                next_state = env.reset()
                model.reset_hidden()

            next_state = torch.FloatTensor(next_state)
            curiosity_reward = curiosity_rewarder.get_reward(state.unsqueeze(0), action, next_state.unsqueeze(0))
            state = next_state

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            curiosity_rewards.append(curiosity_reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _ = model((state.unsqueeze(0), last_act, sum_reward))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            # print(rewards[i], args.curiosity_weight * curiosity_rewards[i].detach())
            R = args.gamma * R + rewards[i] + args.curiosity_weight * curiosity_rewards[i].detach()
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            # print('lp:', log_probs[i], 'gae:', gae.detach(), 'ent:', entropies[i])
            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_weight * entropies[i]

        curiosity_optimizer.zero_grad()
        curiosity_loss = sum(map(lambda x: x**2, curiosity_rewards)) / len(curiosity_rewards)
        curiosity_loss.backward()
        curiosity_optimizer.step()

        optimizer.zero_grad()
        (policy_loss + args.value_weight * value_loss + model.get_loss()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        with lock:
            ensure_shared_grads(model, shared_model)
            optimizer.step()
        model.reset()


def test_worker(args, shared_model, total_steps, optimizer):
    args.environment.clip_rewards = False
    env = make_env(args.environment)

    log_path = '{}/{}'.format(args.train.experiment_folder, 'log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info("STARTED TRAINING PROCESS {}".format(time.strftime("%Y.%m.%d_%H:%M", time.localtime())))

    model = ActorCritic(env.observation_space.shape, env.action_space.n)
    model = BaseWrapper(model)
    if args.train.use_pixel_control:
        model = PixelControlWrapper(model, args.train.gamma, args.train.pc_coef)
    if args.train.use_reward_prediction:
        model = RewardPredictionWrapper(model)
    model.config = args
    model.eval()

    start_time = time.time()

    reward_history = []
    while True:
        model.load_state_dict(shared_model.state_dict())
        if (len(reward_history) + 1) % args.train.save_frequency == 0:
            save_progress(args, model, optimizer, total_steps.value)
        stats = play_game(model, env)
        reward_history.append(stats['total_reward'])

        log_message = (
                'Time {}, num steps {}, FPS {:.0f}, '+
                'curr episode reward {:.2f}, mean episode reward: {:.2f}, '+
                'mean policy loss {:.2f}, mean value loss {:.2f}, '+
                'mean entropy percentage {:.2f}'
            ).format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
            total_steps.value,
            total_steps.value / (time.time() - start_time),
            stats['total_reward'],
            np.mean(reward_history[-60:]),
            stats['policy_loss'],
            stats['value_loss'],
            stats['entropy']
        )
        if args.train.use_pixel_control:
            log_message += ', pixel control loss %.2f' %stats['pc_loss']
        if args.train.use_reward_prediction:
            log_message += ', reward prediction loss %.2f' %stats['rp_loss']
        print(log_message)
        logging.info(log_message)
        time.sleep(60)
