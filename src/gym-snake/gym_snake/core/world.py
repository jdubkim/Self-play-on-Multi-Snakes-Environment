import numpy as np
import random
import sys
sys.path.append('../')
from enum import Enum
from copy import copy
from gym_snake.core.render import Renderer, RGBifier

"""
Setting 
"""
SCREEN_RESOLUTION = 300


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
    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    # DIRECTIONS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
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
        current_position = start_position

        for i in range(1, start_length):
            current_position = tuple(np.subtract(current_position, self.direction))
            self.snake_body.append(tuple(current_position))

    """
        action is an array of length(action_space). 
        Will get a probability of   each actions. 
    """
    def step(self, action):

        if not self.alive:
            return

        # Take the action with highest probability
        # action = np.argmax(action)
        if action == 0:
            return
        # Only move to left or right

        # if self.DIRECTIONS[self.direction] != self.DIRECTIONS[(action % len(self.DIRECTIONS)) - 1]:
        #     self.direction = self.DIRECTIONS[((action + 2) % len(self.DIRECTIONS)) - 1]

        if action == 1 and self.direction != (-1, 0):
            self.direction = (1, 0)
        elif action == 2 and self.direction != (0, -1):
            self.direction = (0, 1)
        elif action == 3 and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif action == 4 and self.direction != (0, 1):
            self.direction = (0, -1)

        # Remove tail
        tail = self.snake_body[-1]
        self.snake_body = self.snake_body[:-1]
        # print("direction is , ", self.DIRECTIONS[self.direction])
        new_head = tuple(map(sum, zip(self.snake_body[0], self.direction)))
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
    FOOD = 255
    DIRECTIONS = Snake.DIRECTIONS
    ACTIONS = Snake.ACTIONS

    def __init__(self, size, n_snakes, n_food=1, is_competitive=False):

        #   Init a numpy matrix with zeros
        self.size = size
        self.dim = size[0]
        self.world = np.zeros(size)
        self.is_competitive = is_competitive
        self.available_positions = set([(i, j) for i in range(self.size[0]) for j in range(self.size[1])])
        self.snakes = []
        self.foods = []
        self.time_step = 0
        #  self.cumulative_rewards = [0 for _ in range(n_snakes)]

        # Init snakes
        for _ in range(n_snakes):
            snake = self.register_snake()
            self.available_positions = self.available_positions - set(snake.snake_body)

        # Place foods
        for _ in range(n_food):
            self.foods.append(self.place_food())

    def free(self):
        self.world = []
        self.snakes = []
        self.available_positions = []

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

        return food_position

    def get_obs_world(self):

        view_dim = self.dim + 2

        obs = np.full((view_dim, view_dim, 3), 0, dtype='uint8')

        for food in self.foods:
            self.render_food(obs, food)

        for i in range(len(self.snakes)):
            color = self.get_snake_color(i)
            self.render_snake(obs, self.snakes[i], color)

        for i in range(view_dim):
            color = self.get_color('wall')
            obs[i][0] = color
            obs[i][self.dim + 1] = color
            obs[0][i] = color
            obs[self.dim + 1][i] = color

        return obs

    def render_food(self, obs, food):
        obs[food[0] + 1][food[1] + 1] = self.get_color('food')

    def render_snake(self, obs, snake, color):
        if len(snake.snake_body) == 0:
            return

        head = snake.snake_body[0]

        for body in snake.snake_body:
            obs[body[0] + 1][body[1] + 1] = color[0]

        obs[head[0] + 1][head[1] + 1] = color[1]

    def get_color(self, key):

        colors = {'food': [255, 0, 0],
                  'wall': [255, 255, 255],
                  'empty': [0, 0, 0]
                  }

        return colors[key]

    def get_snake_color(self, idx):

        p_colors = {0: [[0, 204, 0], [191, 242, 191]],  # Green
                    1: [[0, 51, 204], [128, 154, 230]], # Blue
                    2: [[204, 0, 119], [230, 128, 188]], # Magenta
                    3: [[119, 0, 204], [188, 128, 230]], # Violet
                    }

        return p_colors[idx]

    def get_ob_for_snake(self, idx):

        view_dim = self.dim + 2

        obs = np.full((view_dim, view_dim, 3), 0, dtype='uint8')

        for food in self.foods:
            self.render_food(obs, food)

        for i in range(len(self.snakes)):
            if (i == idx):
                color = self.get_snake_color(0)
            else:
                color = self.get_snake_color(1)
            self.render_snake(obs, self.snakes[i], color)

        for i in range(view_dim):
            color = self.get_color('wall')
            obs[i][0] = color
            obs[i][self.dim + 1] = color
            obs[0][i] = color
            obs[self.dim + 1][i] = color

        return obs

    '''
    Each agent executes action that they got from the network output.
    '''
    def move_snakes(self, actions):

        reward = 0
        done = False
        for idx, snake in enumerate(self.snakes):
            if len(snake.snake_body) == 0 or not snake.alive:
                break

            if idx == 0:
                if actions[idx] is None:
                    print("actions is none")
                if snake is None:
                    print("snake is none")
                print("snake is ", snake)
                new_snake_head, old_snake_tail = snake.step(actions[idx])
                reward, done = self.get_status(snake, new_snake_head, old_snake_tail)
            else:
                print("actions are ", actions)
                print("n snakes are ", len(self.snakes))
                if actions[idx] is None:
                    print("actions is none")
                if snake is None:
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

    # def get_color(self, state):
    #     # VOID -> BLACK
    #     if state == 0:
    #         return [0, 0, 0]
    #     elif state == 255:
    #         return [255, 0, 0]
    #     elif state == 10:   # Wall
    #         return [0, 255, 0]
    #     else:
    #         print("state is ", state)
    #         snake_id = state // 2
    #         is_head = state % 2
    #
    #         if snake_id not in self.p_colors.keys():
    #             snake_id = 0
    #         if is_head == 0:
    #             return self.p_colors[snake_id].body_color
    #         else:
    #             print("snake id is ", snake_id)
    #             return self.p_colors[snake_id].head_color

    def get_status(self, snake, new_snake_head, old_snake_tail):

        reward = 0
        done = False
        other_snakes = copy(self.snakes)
        other_snakes.remove(snake)
        # if self.world[new_snake_head[0]][new_snake_head[1]] == self.FRUIT:

        # If snake collides to the wall or if snake collides himself
        if (not (0 <= new_snake_head[0] < self.size[0]) or not (0 <= new_snake_head[1] < self.size[1])) \
                or new_snake_head in snake.snake_body[1:] or snake.hunger > 50:
            print("you collided to the wall")

            reward = self.REWARD['dead']
            done = True
            snake.alive = False

            self.available_positions = self.available_positions | set(snake.snake_body)
            if self.is_competitive:
                snake.body_to_fruit(self.world)
                snake.hunger = 0
            else:
                snake.snake_body = []
            # remove snake from player
            # self.snakes.remove(snake)
            # add to available positions

        # If snake collides with other snakes
        elif any(new_snake_head in s.snake_body for s in other_snakes):
            print("you collided with other snakes")
            reward = self.REWARD['dead']  # self.cumulative_rewards[i] +
            done = True
            snake.alive = False
            self.available_positions = self.available_positions | set(snake.snake_body)
            if self.is_competitive:
                snake.body_to_fruit(self.world)
            else:
                # snake.free()
                snake.snake_body = []
            # TODO: Adversarial Environment: If they die, make their bodies into food
            # for body in snake.snake_body:
            #     self.world[body[0], body[1]] = self.FOOD
            # remove snake from player
            # self.snakes.remove(snake)
            # add to available positions
        # If snake eats food
        elif self.world[new_snake_head[0], new_snake_head[1]] == self.FOOD:
            print("you ate delicious food")
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
            print("lol I don't know")
            snake.hunger += 1
            reward = self.REWARD['move']
            done = False

        return reward, done