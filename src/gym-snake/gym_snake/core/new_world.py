import random
from copy import copy
import numpy as np

class Snake:

    def __init__(self, snake_id, start_pos, direction, start_length=3):
        self.snake_id = snake_id
        self.start_pos = start_pos
        self.start_length = start_length
        self.snake_length = start_length

        self.alive = True
        self.hunger = 0
        self.snake_body = [start_pos]
        self.direction = direction
        current_pos = start_pos

        #for i in range(1, start_length):
        #    current_pos = tuple(np.subtract(current_pos, self.direction))
        #    self.snake_body.append(current_pos)

        # print("snake body is ", self.snake_body)

    def step(self, action):
        if len(self.snake_body) == 0:
            return 0

        head = self.snake_body[0]
        new_head = head

        print("action is ", action)

        if action == 1 and self.direction != (-1, 0):
            self.direction = (1, 0)
        elif action == 2 and self.direction != (0, -1):
            self.direction = (0, 1)
        elif action == 3 and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif action == 4 and self.direction != (0, 1):
            self.direction = (0, -1)

        if self.direction != (0, 0):
            new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        else:
            print("direction is 0, 0")

        return new_head

class World:
    REWARD = {'dead': -1, 'move': 0, 'eat': 1}
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    FOOD = 255

    def __init__(self, size, n_snakes, n_fruits, seed, is_competitive=False):

        self.size = size
        self.dim = size[0] # or size[1]
        self.world = np.zeros(size)
        self.np_rand = seed
        self.is_competitive = is_competitive
        self.snakes = []
        self.dead_snakes = []
        self.fruits = []
        self.time_step = 0

        # Initialise snakes
        for i in range(n_snakes):
            self.register_snake(i)

        # Initialise fruits
        for i in range(n_fruits):
            self.fruits.append(self.get_safe_cell())

    def register_snake(self, snake_id):
        pos = self.get_rand_cell()
        # while not pos in self.get_available_pos():
        #     pos = (random.randint(snake_size, self.size[0] - snake_size),
        #                 random.randint(snake_size, self.size[1] - snake_size))

        # direction = self.DIRECTIONS[random.randrange(4)]

        new_snake = Snake(snake_id, pos, (0, 0))
        self.snakes.append(new_snake)

        return new_snake

    def move_snakes(self, actions):

        reward = 0.0
        done = False

        # snake == self.snakes[i]
        for i, snake in enumerate(self.snakes):
            new_snake_head = snake.step(actions[i])
            if i == 0: # for main agent
                reward = self.get_status_fruit(snake, new_snake_head)
            else:
                self.get_status_fruit(snake, new_snake_head)

        for i, snake in enumerate(self.snakes):
            snake.alive = self.get_status_alive(snake)
            if not snake.alive:
                if not snake in self.dead_snakes:
                    self.dead_snakes.append(snake)
            if i == 0:
                done = snake.alive

        return reward, done

    def get_status_alive(self, snake):
        if len(snake.snake_body) == 0:
            return False

        head = snake.snake_body[0]

        if (max(head) > self.dim - 1) or (min(head) < 0):
            snake.snake_body = []
            return False

        other_snakes = copy(self.snakes)
        other_snakes.remove(snake)
        for (s_idx, o_snake) in enumerate(other_snakes):
            if head in o_snake.snake_body:
                snake.snake_body = []
                print("in other snakes")
                return False

        if head in snake.snake_body[1:]:
            return False

        return True

    def get_status_fruit(self, snake, new_snake_head):

        if new_snake_head == 0:
            return 0

        reward = 0.0

        eaten_fruits = []

        for i, fruit in enumerate(self.fruits):
            if new_snake_head == fruit:
                eaten_fruits.append(i)
                reward += 1.0
                snake.snake_length += 2

            if len(snake.snake_body) >= snake.snake_length:
                snake.snake_body.pop()

            # print("new snake head is ", new_snake_head)
        snake.snake_body.insert(0, new_snake_head)

        for new_fruit_index in eaten_fruits:
            self.fruits[new_fruit_index] = self.get_safe_cell()

        return reward

    def get_obs_for_snake(self, idx):

        view_dim = self.dim + 2

        obs = np.full((view_dim, view_dim, 3), 0, dtype='uint8')

        for fruit in self.fruits:
            self.render_fruit(obs, fruit)

        for i, snake in enumerate(self.snakes):
            if i == idx:
                color = Color.get_snake_color(0)
            else:
                color = Color.get_snake_color(1)
            self.render_snake(obs, self.snakes[i], color)

        for i in range(view_dim):
            color = Color.get_color('wall')
            obs[i][0] = color
            obs[i][self.dim + 1] = color
            obs[0][i] = color
            obs[self.dim + 1][i] = color

        return obs

    def get_obs_world(self):
        view_dim = self.dim + 2

        obs = np.full((view_dim, view_dim, 3), 0, dtype='uint8')

        for fruit in self.fruits:
            self.render_fruit(obs, fruit)

        for i, snake in enumerate(self.snakes):
            color = Color.get_snake_color(i)
            self.render_snake(obs, snake, color)

        for i in range(view_dim):
            color = Color.get_color('wall')
            obs[i][0] = color
            obs[i][self.dim + 1] = color
            obs[0][i] = color
            obs[self.dim + 1][i] = color

        return obs

    def get_multi_snake_obs(self):

        total_obs = []
        for i, snake in enumerate(self.snakes):
            total_obs.append(self.get_obs_for_snake(i))

        total_obs = np.concatenate(total_obs, axis=2)

        return total_obs
        #t = np.concatenate((self.get_obs_for_snake(0), self.get_obs_for_snake(1)), axis=2)
        #return t  # concatenate two arrays (12 * 12 * 3) convert it to (12 * 12 * 9)

    def render_snake(self, obs, snake, color):
        if len(snake.snake_body) == 0 or not snake.alive:
            return

        head = snake.snake_body[0]

        for body in snake.snake_body:
            obs[body[0] + 1][body[1] + 1] = color[0]

        obs[head[0] + 1][head[1] + 1] = color[1]

    def render_fruit(self, obs, fruit):
        obs[fruit[0] + 1][fruit[1] + 1] = Color.get_color('fruit')

    def get_rand_cell(self):
        return self.np_rand.randint(self.dim), self.np_rand.randint(self.dim)

    def get_safe_cell(self):
        available_pos = list(range(self.dim * self.dim))

        for snake in self.snakes:
            if len(snake.snake_body) > 0:
                used_cells = list(map(lambda x: x[1] * self.dim + x[0], snake.snake_body))
                available_pos = np.setdiff1d(available_pos, used_cells)

        x = 0
        if len(available_pos) > 0:
            x = available_pos[self.np_rand.randint(len(available_pos))]

        return x % self.dim, x // self.dim

    def get_available_pos(self):
        available_pos = set([(i, j) for i in range(self.size[0]) for j in range(self.size[1])])
        for snake in self.snakes:
            available_pos = available_pos - set(snake.snake_body)

        return available_pos

class Color:
    def get_color(key):

        colors = {'fruit': [255, 0, 0],
                  'wall': [255, 255, 255],
                  'empty': [0, 0, 0]
                  }

        return colors[key]

    def get_snake_color(idx):

        p_colors = {0: [[0, 204, 0], [191, 242, 191]],  # Green
                    1: [[0, 51, 204], [128, 154, 230]], # Blue
                    2: [[204, 0, 119], [230, 128, 188]], # Magenta
                    3: [[119, 0, 204], [188, 128, 230]], # Violet
                    }

        return p_colors[idx]



