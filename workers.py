import logging
import numpy as np
import time
import torch
import torch.nn.functional as F

from envs.utils import make_env
from models.actor_critic_rnn import ActorCriticRNN as ActorCritic
from utils import ensure_shared_grads, play_game, save_progress, delta_image


def train_worker(args, shared_model, total_steps, optimizer, lock):
    env = make_env(args)
    args = args.train

    model = ActorCritic(env.observation_space.shape, env.action_space.n)
    model.train()

    state = env.reset()
    state = torch.FloatTensor(state)
    prev_state = None

    while True:
        model.load_state_dict(shared_model.state_dict())
        model.detach_hidden()

        values = []
        log_probs = []
        rewards = []
        entropies = []
        Q_auxes = []

        for step in range(args.update_agent_frequency):
            value, logit, Q_aux = model(state.unsqueeze(0))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            print(state.shape)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            prev_state = state.copy()
            state, reward, done, _ = env.step(action.numpy())
            Q_auxes.append(delta_image(state, prev_state, zoom=4))

            with total_steps.get_lock():
                total_steps.value += 1

            if done:
                state = env.reset()
                model.reset_hidden()

            state = torch.FloatTensor(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _ = model(state.unsqueeze(0))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_weight * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_weight * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        with lock:
            ensure_shared_grads(model, shared_model)
            optimizer.step()


def test_worker(args, shared_model, total_steps, optimizer):
    args.environment.clip_rewards = False
    env = make_env(args)
    
    logging.basicConfig(filename=args.train.logs_path, level=logging.INFO)
    logging.info("STARTED TRAINING PROCESS {}".format(time.strftime("%Y.%m.%d_%H:%M", time.localtime())))

    model = ActorCritic(env.observation_space.shape, env.action_space.n)
    model.eval()

    start_time = time.time()

    reward_history = []
    while True:
        model.load_state_dict(shared_model.state_dict())
        if (len(reward_history) + 1) % args.train.save_frequency == 0:
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
