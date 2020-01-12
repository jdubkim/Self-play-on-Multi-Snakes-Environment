import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
from gym_snake.core.new_world import Snake, World


class NewMultipleSnakes(gym.Env):
    def __init__(self, size=(10, 10), n_snakes=2, n_fruits=4, screen_res=300):
        self.SIZE = size
        self.dim = size[0]
        self.current_step = 0
        self.n_snakes = n_snakes
        self.n_fruits = n_fruits
        self.screen_res = screen_res
        self.seed()
        # Create the world
        self.world = World(size, n_snakes=self.n_snakes, n_fruits=self.n_fruits, seed=self.np_rand)
        self.action_space = spaces.Discrete(5)
        self.viewer = None

    def seed(self, seed=None):
        self.np_rand, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_step = 0
        # Create world
        # self.world.free()
        self.world = World(self.SIZE, n_snakes=self.n_snakes, n_fruits=self.n_fruits, seed=self.np_rand)
        self.steps_beyond_done = None
        return self.world.get_multi_snake_obs()

    def step(self, actions):

        reward, done = self.world.move_snakes(actions)

        if done:
            reward = -1

        self.current_step += 1
        done = self.current_step >= 2000 or done

        n_alives = len(self.world.snakes) - len(self.world.dead_snakes)

        # print("n snakes is ", self.world.snakes)
        # print("dead snakes is ", self.world.dead_snakes)

        return self.world.get_multi_snake_obs(), reward, done, {"ale.lives": 1, "num_snakes": n_alives}

    def render(self, mode='human'):
        dim = self.dim
        screen_dim = self.screen_res
        view_dim = dim + 2
        grid_res = screen_dim / view_dim

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_dim, screen_dim)

            self.grids = []

            for i in range(view_dim):
                for j in range(view_dim):
                    l = grid_res * i
                    r = grid_res * (i + 1)
                    b = grid_res * j
                    t = grid_res * (j + 1)
                    grid = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(grid)

                    self.grids.append(grid)

        obs = self.world.get_obs_world()

        for i in range(view_dim):
            for j in range(view_dim):
                ik = i * view_dim + j
                pixel = obs[i][j]
                self.grids[ik].set_color(pixel[0]/255, pixel[1]/255, pixel[2]/255)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()