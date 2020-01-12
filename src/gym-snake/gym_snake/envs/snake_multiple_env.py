import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
from gym_snake.core.world import Snake, World

"""
Variables:
"""
SCREEN_RESOLUTION = 300

class MultipleSnakes(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'observation.types': ['raw', 'rgb']
    }

    def __init__(self, size=(10, 10), n_food=4, n_snakes=2, step_limit=1000):
        self.SIZE = size
        self.dim = size[0]
        self.STEP_LIMIT = step_limit
        self.current_step = 0
        self.n_snakes = n_snakes
        print("initial n snakes is ", n_snakes)
        self.n_food = n_food

        # Create the world
        self.world = World(size, n_snakes=self.n_snakes, n_food=self.n_food)
        # Set action space 4 directions
        self.action_space = spaces.Discrete(len(self.world.DIRECTIONS))
        self.viewer = None

        self.seed()

    def reset(self):
        self.current_step = 0
        self.alive = True
        # Create world
        # self.world.free()
        self.world = World(self.SIZE, n_snakes=self.n_snakes, n_food=self.n_food)
        return self.world.get_multi_snake_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        # Check whether action is not list or not.
        if not hasattr(actions, '__len__'):
            actions = [actions]

        reward, done = self.world.move_snakes(actions)

        dead_snakes = []

        for i, snake in enumerate(self.world.snakes):
            if not snake.alive:
                dead_snakes.append(snake)

        for snake in dead_snakes:
            snake.free()

        is_main_dead = self.world.snakes[0].alive

        if is_main_dead:
            reward -= 1

        self.current_step += 1

        return self.world.get_multi_snake_obs(), reward, done, \
               {"survived_snakes": len(self.world.snakes) - len(dead_snakes)}

    def render(self, mode='human', close=False):

        if self.world is None:
            return 0

        dim = self.dim
        screen_dim = SCREEN_RESOLUTION

        view_dim = dim + 2
        grid_res = screen_dim / view_dim

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_dim, screen_dim)

            self.grids = []

            for i in range(view_dim):
                for j in range(view_dim):
                    l, r, t, b = grid_res * i, grid_res * (i + 1), grid_res * (j + 1), grid_res * j
                    grid = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(grid)

                    self.grids.append(grid)

        obs = self.world.get_obs_world()

        for i in range(view_dim):
            for j in range(view_dim):
                ik = i * view_dim + j
                pixel = obs[i][j]
                self.grids[ik].set_color(pixel[0]/255, pixel[1]/255, pixel[2]/255)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer: self.viewer.close()