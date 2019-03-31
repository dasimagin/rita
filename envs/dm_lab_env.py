import numpy as np
import deepmind_lab
from gym import spaces
import gym

def _action(*entries):
      return np.array(entries, dtype=np.intc)

ACTIONS = [
    [
        _action(0, 0, 0, 0, 0, 0, 0),
        _action(-15, 0, 0, 0, 0, 0, 0),
        _action(15, 0, 0, 0, 0, 0, 0)
    ],
    [
        _action(0, 0, 0, 0, 0, 0, 0),
        _action(0, 0, -1, 0, 0, 0, 0),
        _action(0, 0, 1, 0, 0, 0, 0)
    ],
    [
        _action(0, 0, 0, 0, 0, 0, 0),
        _action(0, 0, 0, -1, 0, 0, 0),
        _action(0, 0, 0, 1, 0, 0, 0)
    ]
]

class MultiDiscrete:
    def __init__(self):
        self.nvec = np.asarray(
            [3, 3, 3],
            dtype=np.int64
        )
        self.n = self.nvec.prod()

    def __getitem__(self, ind):
        if type(ind) == np.ndarray:
            ind = ind[0][0]
        res = _action(0, 0, 0, 0, 0, 0, 0)
        prod = 1
        for i, ax in enumerate(self.nvec):
            res += ACTIONS[i][int(ind/prod)%ax]
            prod *= ax
        return res

class Dmlab_env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 35
    }

    def __init__(self, args):
        level_config = {
            'fps': str(args.fps),
            'width': str(args.frame_w),
            'height': str(args.frame_h)
        }
        self.env = deepmind_lab.Lab(
            args.env_name,
            ['RGB_INTERLEAVED'],
            config=level_config
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=(args.frame_h, args.frame_w, 3)
        )
        args.id = args.env_name
        self.spec = args
        self.action_space = MultiDiscrete()
        self.reward_range = (-float('inf'), float('inf'))
        self.seed_n = None
        self.np_random = np.random.RandomState()
        self.reset()

    def step(self, action):
        act = self.action_space[action]
        reward = self.env.step(act, num_steps=1)
        done = not self.env.is_running()
        if not done:
            obs = self.env.observations()['RGB_INTERLEAVED']
        else:
            obs = np.zeros(self.observation_space.shape)
        return obs, reward, done, {}

    def reset(self):
        self.env.reset(seed=self.seed_n)
        return self.env.observations()['RGB_INTERLEAVED']

    def render(self, mode=None):
        if self.env.is_running():
            return self.env.observations()['RGB_INTERLEAVED']
        else:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.seed_n = seed

    def get_action_meanings(self):
        return ['NOOP']

    @property
    def unwrapped(self):
        return self
