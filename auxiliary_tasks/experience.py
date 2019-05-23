import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from auxiliary_tasks.agent_wrapper import AgentWrapper
from auxiliary_tasks.pixel_control import pc_reward
from collections import deque

class Transaction:
    def __init__(self, prev_state, state, action,
            reward, terminal, last_action, last_reward, sum_reward):
        if type(prev_state) is not np.ndarray:
            prev_state = prev_state.detach().numpy()
        if type(state) is not np.ndarray:
            state = state.detach().numpy()
        self.state = prev_state
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.pixel_change = pc_reward(prev_state, state)
        self.last_action = last_action
        self.last_reward = last_reward
        self.sum_reward = sum_reward

    def get_last_action_reward(self, action_size):
        return Transaction.concat_action_and_reward(self.last_action, action_size,
                                                        self.sum_reward)

    def get_action_reward(self, action_size):
        return Transaction.concat_action_and_reward(self.action, action_size,
                                                        self.sum_reward+self.reward)

    @staticmethod
    def concat_action_and_reward(action, action_size, reward):
        action_reward = np.zeros([action_size + 1], dtype=np.float32)
        action_reward[action] = 1.0
        action_reward[-1] = float(reward)
        return action_reward


class ExperienceWrapper(AgentWrapper):
    def __init__(self, agent, size=2000):
        super(ExperienceWrapper, self).__init__(agent)
        self._size = size
        self._frames = deque(maxlen=size)
        self._zero_reward = deque()
        self._non_zero_reward = deque()
        self._top_frame_index = 0

    def add_frame(self, frame):
        if frame.terminal and len(self._frames) > 0 and self._frames[-1].terminal:
            return

        frame_index = self._top_frame_index + len(self._frames)
        was_full = self.is_full()

        self._frames.append(frame)

        if frame_index >= 3:
            if frame.reward == 0:
                self._zero_reward.append(frame_index)
            else:
                self._non_zero_reward.append(frame_index)

        if was_full:
            self._top_frame_index += 1

            cut_frame_index = self._top_frame_index + 3
            if len(self._zero_reward) > 0 and \
                    self._zero_reward[0] < cut_frame_index:
                self._zero_reward.popleft()

            if len(self._non_zero_reward) > 0 and \
                    self._non_zero_reward[0] < cut_frame_index:
                self._non_zero_reward.popleft()

    def is_full(self):
        return len(self._frames) >= self._size

    def sample_sequence(self, sequence_size):
        start_pos = np.random.randint(0, len(self._frames) - sequence_size - 1)

        if self._frames[start_pos].terminal:
            start_pos += 1

        sampled_frames = []

        for i in range(sequence_size):
            frame = self._frames[start_pos + i]
            sampled_frames.append(frame)
            if frame.terminal:
                break

        return sampled_frames

    def sample_rp_sequence(self):
        if np.random.randint(2) == 0:
            from_zero = True
        else:
            from_zero = False

        if len(self._zero_reward) == 0:
            from_zero = False
        elif len(self._non_zero_reward) == 0:
            from_zero = True

        if from_zero:
            index = np.random.randint(len(self._zero_reward))
            end_frame_index = self._zero_reward[index]
        else:
            index = np.random.randint(len(self._non_zero_reward))
            end_frame_index = self._non_zero_reward[index]

        start_frame_index = end_frame_index - 3
        raw_start_frame_index = start_frame_index - self._top_frame_index

        sampled_frames = []

        for i in range(4):
            frame = self._frames[raw_start_frame_index + i]
            sampled_frames.append(frame)

        return sampled_frames
