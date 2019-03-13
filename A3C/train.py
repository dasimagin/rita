import torch
import torch.nn.functional as F
import torch.optim as optim

from openai_wrappers import make_atari
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            shared_param._grad += param.grad
        else:
            shared_param._grad = param.grad


def train(args, shared_model, total_steps, optimizer, lock):
    env = make_atari(args.env_name)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.train()

    state = env.reset()
    state = torch.FloatTensor(state)

    while True:
        model.load_state_dict(shared_model.state_dict())
        model.detach_hidden()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit = model(state.unsqueeze(0))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.numpy())

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

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        with lock:
            ensure_shared_grads(model, shared_model)
            optimizer.step()
