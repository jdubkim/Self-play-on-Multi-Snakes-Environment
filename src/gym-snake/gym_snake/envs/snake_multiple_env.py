import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
from config import Config
from gym_snake.core.world import Snake, World

class MultipleSnakes(gym.Env):
    COLOR_CHANNELS = 3
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'observation.types': ['raw', 'rgb']
    }

    def __init__(self, size=(10, 10), step_limit=1000, n_food=4, n_snakes=2):
        self.SIZE = size
        self.dim = size[0]
        self.STEP_LIMIT = step_limit
        self.hunger = 0
        self.current_step = 0
        self.alive = True
        self.n_snakes = n_snakes
        # Create the world
        self.world = World(size, n_snakes=self.n_snakes, n_food=n_food)
        # Set action space 4 directions
        # TODO: Add one more action space for 'attack'
        self.action_space = spaces.Discrete(len(self.world.DIRECTIONS) + 1)
        self.viewer = None
        self.seed()
        # Set renderer

    def reset(self):
        self.current_step = 0
        self.alive = True
        # Create world
        self.world = World(self.SIZE, n_snakes=self.n_snakes)
        return self.world.get_multi_snake_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        # Check whether action is not list or not.
        if not hasattr(actions, '__len__'):
            actions = [actions]

        reward, done = self.world.move_snakes(actions)

        if not self.alive:
            raise Exception('Need to reset env now.')

        self.current_step += 1

        return self.world.get_multi_snake_obs(), reward, done, {}

    def render(self, mode='human', close=False):
        return self.world.render(mode, close)

    def close(self):
        if self.viewer: self.viewer.close()