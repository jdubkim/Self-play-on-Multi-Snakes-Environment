import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np

from gym_snake.core.render import Renderer, RGBifier
from gym_snake.core.world import World

class SingleSnake(gym.Env):
    COLOR_CHANNELS = 3
    metadata = {
        'render.modes': ['human','rgb_array'],
        'observation.types': ['raw', 'rgb']
    }

    def __init__(self, size=(20, 20), step_limit=1000, dynamic_step_limit = 1000, obs_type='rgb',
                 obs_zoom=1, n_food=4, render_zoom=5):
        self.SIZE = size
        self.STEP_LIMIT = step_limit
        self.DYNAMIC_STEP_LIMIT = dynamic_step_limit
        self.hunger = 0
        self.current_step = 0
        self.n_food = n_food
        # Create the world
        self.world = World(size, n_snakes=1, n_food=self.n_food)

        self.obs_type = obs_type
        if self.obs_type == 'raw':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.SIZE[0], self.SIZE[1]))
        elif self.obs_type == 'rgb':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.SIZE[0], self.SIZE[1], self.COLOR_CHANNELS))
            self.RGBify = RGBifier(self.SIZE, zoom_factor=obs_zoom, players_colors={})
        else:
            raise(Exception('Unrecognized observation mode.'))

        # Set action space 4 directions
        self.action_space = spaces.Discrete(len(self.world.DIRECTIONS))
        # Set renderer
        self.RENDER_ZOOM = render_zoom

    def reset(self):
        self.current_step = 0
        self.alive = True
        self.hunger = 0
        # Create world
        self.world = World(self.SIZE, n_snakes=1, n_food = self.n_food)
        return self.get_state()

    def step(self, action):
        if not self.alive:
            raise Exception('Need to reset env now.')

        self.current_step += 1
        if (self.current_step >= self.STEP_LIMIT) or (self.hunger > self.DYNAMIC_STEP_LIMIT):
            self.alive = False
            # return observation, alive,
            return self.world.get_observation(), 0, True, {}  # use {} to pass information

        rewards, dones = self.world.move_snake(np.array([action]))
        # Update
        self.hunger += 1
        if rewards[0] > 0:
            self.hunger = 0

        if dones[0]:
            self.alive = False

        return self.get_state(), rewards[0], dones[0], {}

    def seed(self, seed):
        return random.seed(seed)

    def get_state(self):
        state = self.world.get_observation()

        return self.RGBify.get_image(state)

    def render(self, mode='rgb_array', close=False):
        if not close:
            if not hasattr(self, 'renderer'):
                self.renderer = Renderer(self.SIZE, zoom_factor=self.RENDER_ZOOM, players_colors={})
            return self.renderer.render(self.world.get_observation(), mode=mode, close=close)