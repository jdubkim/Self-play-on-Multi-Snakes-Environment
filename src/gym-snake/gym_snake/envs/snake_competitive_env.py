import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random

from gym_snake.core.render import Renderer, RGBifier
from gym_snake.core.world import World


class CompetitiveSnakes(gym.Env):

    COLOR_CHANNELS = 3
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'observation.types': ['raw', 'rgb']
    }

    def __init__(self, size=(24, 24), step_limit=1000, dynamic_step_limit=1000, obs_type='rgb',
                 obs_zoom=1, n_food=4, n_snakes=4, render_zoom=20):
        self.SIZE = size
        self.STEP_LIMIT = step_limit
        self.DYNAMIC_STEP_LIMIT = dynamic_step_limit
        self.hunger = 0
        self.current_step = 0
        self.alive = True
        self.n_snakes = n_snakes
        self.n_food = n_food
        # Create the world
        self.world = World(size, n_snakes=self.n_snakes, n_food=n_food, is_competitive=True)

        self.obs_type = obs_type
        if self.obs_type == 'raw':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.SIZE[0], self.SIZE[1]))
        elif self.obs_type == 'rgb':
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.SIZE[0], self.SIZE[1], self.COLOR_CHANNELS))
            self.RGBify = RGBifier(self.SIZE, zoom_factor=obs_zoom, players_colors={})
        else:
            raise(Exception('Unrecognized observation mode.'))

        # Set action space 4 directions
        # TODO: Add one more action space for 'attack'
        self.action_space = spaces.Discrete(len(self.world.DIRECTIONS))
        # Set renderer
        self.RENDER_ZOOM = render_zoom

    def _reset(self):
        self.current_step = 0
        self.alive = True
        # Create world
        self.world = World(self.SIZE, n_snakes=self.n_snakes, n_food=self.n_food, is_competitive=True)
        return self._get_state()

    def _step(self, actions):

        if not self.alive:
            raise Exception('Need to reset env now.')

        rewards, dones = self.world.move_snake(actions)
        self.alive = not all(dones)

        self.current_step += 1
        if self.current_step >= self.STEP_LIMIT:
            self.alive = False
            # return observation, alive,
            return self.world.get_observation(), rewards, dones, {}

        for i, (reward, done) in enumerate(zip(rewards, dones)):
            if reward > 0:
                self.world.snakes[i].hunger = 0

        return self._get_state(), rewards, dones, {}

    def _get_state(self):
        state = self.world.get_observation()
        if self.obs_type == 'rgb':
            return self.RGBify.get_image(state)
        else:
            return state

    def _render(self, mode='human', close=False):
        if not close:
            if not hasattr(self, 'renderer'):
                self.renderer = Renderer(self.SIZE, zoom_factor=self.RENDER_ZOOM, players_colors={})
            return self.renderer.render(self.world.get_observation(), mode=mode, close=close)