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

    def __init__(self, size=(10, 10), step_limit=1000, dynamic_step_limit=1000,
                 obs_zoom=4, n_food=4, render_zoom=4):
        self.SIZE = size
        self.view_dim = (size[0] + 2, size[1] + 2)
        self.STEP_LIMIT = step_limit
        self.DYNAMIC_STEP_LIMIT = dynamic_step_limit
        self.hunger = 0
        self.current_step = 0
        self.n_food = n_food
        # Create the world
        self.world = World(size, n_snakes=2, n_food=self.n_food)

        # self.observation_space = spaces.Box(low=0, high=255, shape=(self.view_dim[0] * obs_zoom, self.view_dim[1] * obs_zoom, self.COLOR_CHANNELS), dtype='uint8')
        self.RGBify = RGBifier(self.view_dim, zoom_factor=obs_zoom, players_colors={})

        # Set action space 4 directions
        self.action_space = spaces.Discrete(len(self.world.DIRECTIONS))
        # Set renderer
        self.RENDER_ZOOM = render_zoom

    def reset(self):
        self.current_step = 0
        self.alive = True
        self.hunger = 0
        # Create world
        self.world = World(self.SIZE, n_snakes=2, n_food=self.n_food)
        return self.get_state()

    def step(self, action):
        if not self.alive:
            raise Exception('Need to reset env now.')

        self.current_step += 1
        if (self.current_step >= self.STEP_LIMIT) or (self.hunger > self.DYNAMIC_STEP_LIMIT):
            self.alive = False
            # return observation, alive,
            return self.get_state(), 0, True, {}  # use {} to pass information

        rewards, dones = self.world.move_snake(np.array([action]))
        # Update
        self.hunger += 1
        if rewards[0] > 0:
            self.hunger = 0

        if dones[0]:
            self.alive = False

        return self.get_state(), rewards, dones, {} # {"ale.lives": 1, "num_snakes": (len(snakes) - len(dead_idxs))}

    def seed(self, seed):
        return random.seed(seed)

    def get_state(self):
        state = self.world.get_observation_total()
        # image = self.RGBify.get_img_array(state)
        return state

    def render(self, mode='rgb_array', close=False):
        if not close:
            if not hasattr(self, 'renderer'):
                self.renderer = Renderer(self.view_dim, zoom_factor=self.RENDER_ZOOM, players_colors={})
            return self.renderer.render(self.world.get_observation_world(), mode=mode, close=close)