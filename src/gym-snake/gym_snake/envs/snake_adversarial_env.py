import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
from config import Config

class SnakeAdversarial(gym.Env):
    def __init__(self):
        self.dim = 10  # 10 X 10 environment
        self.action_space = spaces.Discrete(5)
        self.viewer = None
        self.spare_fruits = 0

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_snake(self, ob, snake, body_color, head_color):
        if (len(snake) == 0):
            return

        head = snake[0]

        for piece in snake:
            ob[piece[0] + 1][piece[1] + 1] = body_color

        ob[head[0] + 1][head[1] + 1] = head_color

    def get_ob_for_snake(self, idx):
        dim = self.dim

        ob = np.full((dim + 2, dim + 2, 3), 0, dtype='uint8')

        snakes = self.state[0]
        fruits = self.state[1]

        for fruit in fruits:
            ob[fruit[0] + 1][fruit[1] + 1] = [255, 0, 0]

        for i in range(len(snakes)):
            if (i == idx):
                self.draw_snake(ob, snakes[i], [0, 204, 0], [191, 242, 191])
            else:
                self.draw_snake(ob, snakes[i], [0, 51, 204], [128, 154, 230])

        for i in range(dim + 2):
            ob[i][0] = [255, 255, 255]
            ob[i][dim + 1] = [255, 255, 255]
            ob[0][i] = [255, 255, 255]
            ob[dim + 1][i] = [255, 255, 255]

        return ob  # 12 * 12 * 3 array

    def get_ob_world(self):
        dim = self.dim

        ob = np.full((dim + 2, dim + 2, 3), 0, dtype='uint8')

        snakes = self.state[0]
        fruits = self.state[1]

        for fruit in fruits:
            ob[fruit[0] + 1][fruit[1] + 1] = [255, 0, 0]

        for i in range(len(snakes)):
            color = self.get_color(i)
            self.draw_snake(ob, snakes[i], color[0], color[1])

        for i in range(dim + 2):
            ob[i][0] = [255, 255, 255]
            ob[i][dim + 1] = [255, 255, 255]
            ob[0][i] = [255, 255, 255]
            ob[dim + 1][i] = [255, 255, 255]

        return ob  # 12 * 12 * 3 array

    def get_color(self, idx):

        p_colors = {0: [[0, 204, 0], [191, 242, 191]],  # Green
                    1: [[0, 51, 204], [128, 154, 230]], # Blue
                    2: [[204, 0, 119], [230, 128, 188]], # Magenta
                    3: [[119, 0, 204], [188, 128, 230]], # Violet
                    }

        return p_colors[idx]

    def get_multi_snake_ob(self):
        t = np.concatenate((self.get_ob_for_snake(0), self.get_ob_for_snake(1), self.get_ob_for_snake(2)), axis=2)
        return t  # concatenate two arrays (12 * 12 * 3) convert it to (12 * 12 * 9)

    def update_snake(self, idx, action):
        [snakes, fruits, vels, grow_to_lengths, t] = self.state

        snake = snakes[idx]

        if (len(snake) == 0):
            return 0

        head = snake[0]
        vel = vels[idx]

        if (action == 1 and vel != (-1, 0)):
            vel = (1, 0)
        elif (action == 2 and vel != (0, -1)):
            vel = (0, 1)
        elif (action == 3 and vel != (1, 0)):
            vel = (-1, 0)
        elif (action == 4 and vel != (0, 1)):
            vel = (0, -1)

        reward = 0.0

        if vel != (0, 0):
            head = (head[0] + vel[0], head[1] + vel[1])

            snake_length = grow_to_lengths[idx]

            pending_fruit = []

            for i in range(len(fruits)):
                fruit = fruits[i]

                if head == fruit:
                    pending_fruit.append(i)
                    reward += 1.0
                    snake_length += 2

            if len(snake) >= snake_length:
                snake.pop()

            snake.insert(0, head)

            for i in pending_fruit:
                if self.spare_fruits > 0:
                    self.spare_fruits -= 1
                elif self.spare_fruits == 0:
                    fruits[i] = self.safe_choose_cell()
            vels[idx] = vel
            grow_to_lengths[idx] = snake_length

        return reward

    def is_snake_alive(self, idx):
        snakes = self.state[0]
        snake = snakes[idx]

        if (len(snake) == 0):
            return False

        head = snake[0]

        if (max(head) > self.dim - 1) or (min(head) < 0):
            return False

        for (s_idx, snake) in enumerate(snakes):
            for i in range(0, len(snake)):
                if head == snake[i] and not (i == 0 and s_idx == idx):
                    return False

        return True

    def step(self, action):
        if not hasattr(action, '__len__'):
            action = [action]

        [snakes, fruits, vels, grow_to_lengths, t] = self.state

        reward = self.update_snake(0, action[0])

        for idx in range(1, len(snakes)):
            action_other = action[idx]
            self.update_snake(idx, action_other)

        dead_idxs = []

        for idx in range(len(snakes)):
            if not self.is_snake_alive(idx):
                dead_idxs.append(idx)
                # body becomes fruits
                for body in snakes[idx]:
                    fruits.append(body)
                    self.spare_fruits += len(snakes[idx])

        for idx in dead_idxs:
            snakes[idx] = []

        isMainDead = len(snakes[0]) == 0

        if isMainDead:
            reward = -1

        t += 1
        self.state[4] = t

        done = t >= 2000 or isMainDead

        return self.get_multi_snake_ob(), reward, done, {"ale.lives": 1, "num_snakes": (len(snakes) - len(dead_idxs))}

    def choose_cell(self):
        return (self.np_random.randint(self.dim), self.np_random.randint(self.dim))

    def safe_choose_cell(self):
        snakes = self.state[0]

        available = list(range(self.dim * self.dim))

        for snake in snakes:
            if len(snake) > 0:
                used_idxs = list(map(lambda x: x[1] * self.dim + x[0], snake))
                available = np.setdiff1d(available, used_idxs)

        x = 0

        if len(available) > 0:
            x = available[self.np_random.randint(len(available))]

        return (x % self.dim, x // self.dim)

    def reset(self):
        snakes = []
        fruits = []

        for i in range(Config.NUM_SNAKES):
            snakes.append([self.choose_cell()])
            fruits.append(self.choose_cell())

        grow_to_lengths = [3, 3, 3]
        vels = [(0, 0), (0, 0), (0, 0)]

        self.state = [snakes, fruits, vels, grow_to_lengths, 0]
        self.steps_beyond_done = None
        return self.get_multi_snake_ob()

    def render(self, mode='human'):
        dim = self.dim
        screen_dim = 300

        head_cell = self.state[0]
        fruit_cell = self.state[1]

        view_dim = dim + 2
        cell_dim = screen_dim / view_dim

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_dim, screen_dim)

            cells = []

            for i in range(view_dim):
                for j in range(view_dim):
                    l, r, t, b = cell_dim * i, cell_dim * (i + 1), cell_dim * (j + 1), cell_dim * j,
                    cell = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(cell)

                    cells.append(cell)

            self.cells = cells

        ob = self.get_ob_world()

        for i in range(view_dim):
            for j in range(view_dim):
                idx = i * view_dim + j
                rgb255 = ob[i][j]
                self.cells[idx].set_color(rgb255[0] / 255, rgb255[1] / 255, rgb255[2] / 255)

        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
