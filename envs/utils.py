import gym

from envs.common_wrappers import NoopResetEnv, ClipRewardEnv, MaxAndSkipEnv, ProcessFrame84
from envs.common_wrappers import FrameStack, ExtraTimeLimit, ImageToPyTorch, NormalizedEnv, RescaleImageEnv
from envs.dm_lab_env import Dmlab_env

def make_atari(args):
    env = gym.make(args.env_name)
    env = NoopResetEnv(env, noop_max=args.noop_max)
    if args.clip_rewards:
        env = ClipRewardEnv(env)
    if 'NoFrameskip' in env.spec.id and args.skip_frames > 0:
        env = MaxAndSkipEnv(env, skip=args.skip_frames)
    env = ProcessFrame84(env, crop=False)
    if args.stack_frames > 1:
        env = FrameStack(env, args.stack_frames)
    env = ExtraTimeLimit(env, args.max_episode_steps)
    env = ImageToPyTorch(env)
    if args.normalize_env:
        env = NormalizedEnv(env)
    return env

def make_dm_lab(args, recording):
    env = Dmlab_env(args)
    if recording:
        env = RescaleImageEnv(env, size=(args.prev_frame_h, args.prev_frame_w))
    if args.clip_rewards:
        env = ClipRewardEnv(env)
    if args.skip_frames > 0:
        env = MaxAndSkipEnv(env, skip=args.skip_frames)
    if args.stack_frames > 1:
        env = FrameStack(env, args.stack_frames)
    env = ImageToPyTorch(env)
    if args.normalize_env:
        env = NormalizedEnv(env)
    return env

def make_env(env_args, recording=False):
    if env_args.env_type == "atari":
        return make_atari(env_args)
    elif env_args.env_type == "dmlab":
        return make_dm_lab(env_args, recording)
    else:
        raise NotImplemented
