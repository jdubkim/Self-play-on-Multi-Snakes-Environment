import gym
from gym import spaces
import numpy as np
import sys
from baselines.common import set_global_seeds
from baselines.bench import Monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from config import Config

import cv2

cv2.ocl.setUseOpenCL(False)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        if Config.USE_ATARI_SIZE:
            self.width = 84
            self.height = 84
        else:
            self.width = 12
            self.height = 12

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 6), dtype=np.uint8)

    def observation(self, frame):
        if Config.USE_ATARI_SIZE:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return frame


def make_basic_env(env_id, num_env, seed, start_index=0):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, None, allow_early_resets=True)

            env = WarpFrame(env)

            return env

        return _thunk

    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def get_shape(ob_space):
    shape = ob_space.shape
    shape = (shape[0], shape[1], int(shape[2] / 2))

    return shape

def get_opponent1_file(x):
    return 'opponent1_' + str(x) + '.pkl'

def get_opponent2_file(x):
    return 'opponent2_' + str(x) + '.pkl'