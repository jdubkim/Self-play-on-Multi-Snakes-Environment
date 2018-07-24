import numpy as np
import random
import sys
sys.path.append('../')
from enum import Enum
from copy import copy
from gym_snake.core.render import Renderer, RGBifier

"""
    Set snake as an agent in this environment. 
    Initially starts with length = 3, and direction towards North
"""

class Snake:
    """
        Directions:
        0: UP (N)
        1: RIGHT (E)
        2: DOWN (S)
        3: LEFT (W)
        Actions:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT
        4: ATTACK -> to be implemented
    """
    # Directions: [0] null [1] up [2] right [3] down [4] left
    DIRECTIONS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
    ACTIONS = [DIRECTIONS, 'attack']
    # j: left l: down h: up k: right

    def __init__(self, snake_id, start_position, direction=DIRECTIONS[0], start_length=3):
        self.snake_id = snake_id
        self.direction = direction
        self.length = start_length
        self.start_position = start_position

        self.hunger = 0
        self.alive = True
        self.already_dead = False
        self.snake_body = [start_position]  # save coordinates of snake body. array of tuples
        current_position = np.array(start_position)

        for i in range(1, start_length):
            current_position = current_position - self.DIRECTIONS[self.direction]
            self.snake_body.append(tuple(current_position))

        class SnakeColor:

            def __init__(self, head_color, body_color):
                self.head_color = head_color
                self.body_color = body_color

        self.p_colors = {1: SnakeColor([191, 242, 191], [0, 204, 0]),  # Green
                         2: SnakeColor([188, 128, 230], [119, 0, 204]),  # Violet
                         3: SnakeColor([128, 154, 230], [0, 51, 204]),  # Blue
                         4: SnakeColor([230, 128, 188], [204, 0, 119]),  # Magenta
                         10: SnakeColor([255, 255, 189], [255, 255, 0])}  # Opponent, orange

    """
        action is an array of length(action_space). 
        Will get a probability of each actions. 
    """
    def step(self, action):

        action -= 1

        if not self.alive:
            return

        # Take the action with highest probability
        # action = np.argmax(action)

        # Only move to left or right
        if (action != self.direction) and (action != (self.direction + 2) % len(self.DIRECTIONS)):
            self.direction = action
        else:
            action = self.direction

        # Remove tail
        tail = self.snake_body[-1]
        self.snake_body = self.snake_body[:-1]
        print("action is ", action)
        print("self.direction is ", self.direction)
        print("direction is , ", self.DIRECTIONS[self.direction])
        new_head = tuple(np.array(self.snake_body[0]) + self.DIRECTIONS[self.direction])
        print("new head is ", new_head)
        self.snake_body.insert(0, new_head)

        return new_head, tail

    def free(self):
        if self.alive:
            return

        self.hunger = 0
        self.snake_body = []

    def body_to_fruit(self, world):
        FOOD = 255

        for body in self.snake_body[1:]:
            world[body[0], body[1]] = FOOD
        self.snake_body = []


class World:
    REWARD = {'dead': -1, 'move': 0, 'eat': 1}

    def __init__(self, size, n_snakes, n_food=1, is_competitive=False):
        self.FOOD = 255
        self.DIRECTIONS = Snake.DIRECTIONS
        self.ACTIONS = Snake.ACTIONS

        #   Init a numpy matrix with zeros
        self.size = size
        self.dim = size[0]
        self.world = np.zeros(size)
        self.is_competitive = is_competitive
        self.available_positions = set([(i, j) for i in range(self.size[0]) for j in range(self.size[1])])
        self.snakes = []
        self.time_step = 0
        #  self.cumulative_rewards = [0 for _ in range(n_snakes)]

        for _ in range(n_snakes):
            snake = self.register_snake()
            self.available_positions = self.available_positions - set(snake.snake_body)

        for _ in range(n_food):
            self.place_food()

    # snake_id starts from 1
    def register_snake(self):
        SNAKE_SIZE = 4
        # Make snakes not overlap
        position = (random.randint(SNAKE_SIZE, self.size[0] - SNAKE_SIZE), random.randint(SNAKE_SIZE, self.size[1] - SNAKE_SIZE))

        while not position in self.available_positions:
            position = (random.randint(SNAKE_SIZE, self.size[0] - SNAKE_SIZE), random.randint(SNAKE_SIZE, self.size[1] - SNAKE_SIZE))

        direction = random.randrange(len(Snake.DIRECTIONS))
        # create snake
        new_snake = Snake(len(self.snakes) + 1, position, direction, SNAKE_SIZE)
        self.snakes.append(new_snake)

        return new_snake

    def place_food(self):
        # Choose a place
        food_position = random.choice(list(self.available_positions))
        self.world[food_position[0], food_position[1]] = self.FOOD

    def get_obs_world(self):

        obs = self.world.copy()
        obs = np.pad(obs, pad_width=1, mode='constant', constant_values=10)  # set up the wall

        for snake in self.snakes:
            if not snake.alive:
                continue

            for block in snake.snake_body:
                obs[block[0], block[1]] = snake.snake_id * 2
            # Highlight head
            obs[snake.snake_body[0][0], snake.snake_body[0][1]] = snake.snake_id * 2 + 1

        color_lu = np.vectorize(lambda x: self.get_color(x), otypes=[np.uint8, np.uint8, np.uint8])
        obs = np.array(color_lu(obs))

        return obs

    def get_ob_for_snake(self, idx):

        OPPONENT_COLOR = 10

        obs = self.world.copy()
        obs = np.pad(obs, pad_width=1, mode='constant', constant_values=10) # set up the wall
        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                continue

            if idx == i:  # if snake itself
                for block in snake.snake_body:
                    obs[block[0], block[1]] = snake.snake_id * 2
                # Highlight head
                obs[snake.snake_body[0][0], snake.snake_body[0][1]] = snake.snake_id * 2 + 1

            else:  # set other opponents same color
                for block in snake.snake_body:
                    obs[block[0], block[1]] = OPPONENT_COLOR * 2
                    obs[snake.snake_body[0][0], snake.snake_body[0][1]] = OPPONENT_COLOR * 2 + 1

        color_lu = np.vectorize(lambda x: self.get_color(x), otypes=[np.uint8, np.uint8, np.uint8])
        obs = np.array(color_lu(obs))

        return obs  # 12 * 12 * 3 array

    '''
    Each agent executes action that they got from the network output.
    '''
    def move_snakes(self, actions):

        reward = 0
        done = False
        if self.snakes is None:
            print("all snakes are dead")
        for idx, snake in enumerate(self.snakes):
            if idx == 0:
                new_snake_head, old_snake_tail = snake.step(actions[idx])
                reward, done = self.get_status(snake, new_snake_head, old_snake_tail)
            else:
                print("actions are ", actions)
                print("n snakes are ", len(self.snakes))
                if actions[idx] is None:
                    print("actions is none")
                elif snake is None:
                    print("snake is none")
                new_snake_head, old_snake_tail = snake.step(actions[idx])
                self.get_status(snake, new_snake_head, old_snake_tail)

        self.time_step += 1

        done = done or (self.time_step >= 2000)

        return reward, done

    def get_multi_snake_obs(self):

        total_obs = []
        for i, snake in enumerate(self.snakes):
            total_obs.append(self.get_ob_for_snake(i))

        total_obs = np.concatenate(total_obs, axis=2)

        return total_obs

        # t = np.concatenate((self.get_ob_for_snake(0), self.get_ob_for_snake(1)), axis=2)
        # return t  # concatenate two arrays (12 * 12 * 3) convert it to (12 * 12 * 6)

    def get_color(self, state):
        # VOID -> BLACK
        if state == 0:
            return [0, 0, 0]
        elif state == 255:
            return [255, 0, 0]
        elif state == 10:   # Wall
            return [0, 255, 0]
        else:
            print("state is ", state)
            snake_id = state // 2
            is_head = state % 2

            if snake_id not in self.p_colors.keys():
                snake_id = 0
            if is_head == 0:
                return self.p_colors[snake_id].body_color
            else:
                print("snake id is ", snake_id)
                return self.p_colors[snake_id].head_color

    def get_status(self, snake, new_snake_head, old_snake_tail):

        reward = 0
        done = False
        other_snakes = copy(self.snakes)
        other_snakes.remove(snake)

        # If snake collides to the wall or if snake collides himself
        if (not (0 <= new_snake_head[0] < self.size[0]) or not (0 <= new_snake_head[1] < self.size[1])) \
                or new_snake_head in snake.snake_body[1:] or snake.hunger > 50:

            reward = self.REWARD['dead']
            done = True
            snake.alive = False

            if self.is_competitive:
                snake.body_to_fruit(self.world)
                snake.hunger = 0
            else:
                snake.free()
            # remove snake from player
            # self.snakes.remove(snake)
            # add to available positions
            self.available_positions = self.available_positions | set(snake.snake_body)

        # If snake collides with other snakes
        elif any(new_snake_head in s.snake_body for s in other_snakes):
            reward = self.REWARD['dead']  # self.cumulative_rewards[i] +
            done = True
            snake.alive = False
            if self.is_competitive:
                snake.body_to_fruit(self.world)
            else:
                snake.free()
            # TODO: Adversarial Environment: If they die, make their bodies into food
            # for body in snake.snake_body:
            #     self.world[body[0], body[1]] = self.FOOD
            # remove snake from player
            # self.snakes.remove(snake)
            # add to available positions
            self.available_positions = self.available_positions | set(snake.snake_body)

        # If snake eats food
        elif self.world[new_snake_head[0], new_snake_head[1]] == self.FOOD:
            # Remove food
            self.world[new_snake_head[0], new_snake_head[1]] = 0
            # Add tail
            snake.snake_body.append(old_snake_tail)
            snake.length += 1
            snake.hunger = 0
            # Place new food
            self.place_food()
            reward = self.REWARD['eat']
            done = False
        else:
            snake.hunger += 1
            reward = self.REWARD['move']
            done = False

        return reward, done

    def get_image(self, state):
        print("state shape is ", state.shape)
        screen_dim = 240
        dim = self.size[0] + 2
        cell_dim = (int) (screen_dim / dim)
        print("Size is ", self.size)
        # Transform to RGB image with 3 channels
        COLOR_CHANNELS = 3
        # Zoom every channel
        img_zoomed = np.zeros((3, dim * cell_dim, dim * cell_dim), dtype=np.uint8)

        for c in range(COLOR_CHANNELS):
            for i in range(state.shape[1]):
                for j in range(state.shape[2]):
                    img_zoomed[c, i * cell_dim: (i + 1) * cell_dim,
                    j * cell_dim: (j + 1) * cell_dim] = np.full(
                        (cell_dim, cell_dim), state[c, i, j])
        # Transpose to get channels as last

        return img_zoomed

    def render(self, state, mode='human', close=False):
        if close:
            self.close()
            return

        img = self.get_image(self.get_obs_world())

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
