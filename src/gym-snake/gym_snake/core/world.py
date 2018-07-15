import numpy as np
import random
from enum import Enum
from copy import copy
from operator import add
"""
    Set snake as an agent in this environment. 
    Initially starts with length = 3, and direction towards North
"""

class Dir(Enum):
    W = 0
    E = 1
    N = 2
    S = 3

    def describe(self):
        return self.name, self.value

    def __str__(self):
        return 'direction is {0}'.format(self.name)

    def rotate_right(self):
        return (self.value + 1) % len(Dir)

    def rotate_left(self):
        return (len(Dir) + self.value - 1) % len (Dir)

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
    DIRECTIONS = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
    ACTIONS = [DIRECTIONS, 'attack']

    def __init__(self, snake_id, start_position, direction=Dir.N, start_length=3):
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

    def step(self, action):

        if not self.alive:
            return

        action = np.argmax(action)

        # Only move to left or right
        if (action != self.direction) and (action != (self.direction + 2) % len(self.DIRECTIONS)):
            self.direction = action
        self.hunger += 1
        # Remove tail
        tail = self.snake_body[-1]
        self.snake_body = self.snake_body[:-1]
        new_head = tuple(np.array(self.snake_body[0]) + self.DIRECTIONS[self.direction])
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
    REWARD = {'dead': -5, 'move': 0, 'eat': 1}

    def __init__(self, size, n_snakes, n_food=1, is_competitive=False):
        self.FOOD = 255
        self.DIRECTIONS = Snake.DIRECTIONS
        self.ACTIONS = Snake.ACTIONS

        # Init a numpy matrix with zeros
        self.size = size
        self.world = np.zeros(size)
        self.is_competitive = is_competitive
        self.available_positions = set([(i, j) for i in range(self.size[0]) for j in range(self.size[1])])
        self.snakes = []
        self.cumulative_rewards = [0 for _ in range(n_snakes)]

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

    def get_observation(self):
        obs = self.world.copy()

        for snake in self.snakes:
            if not snake.alive:
                continue

            for block in snake.snake_body:
                obs[block[0], block[1]] = snake.snake_id * 2
            # Highlight head
            obs[snake.snake_body[0][0], snake.snake_body[0][1]] = snake.snake_id * 2 + 1
        return obs

    '''
    Each agent executes action that they got from the network output.
    '''
    def move_snake(self, actions):
        rewards = []
        dones = []

        for i, (snake, action) in enumerate(zip(self.snakes, np.nditer(actions))):
            if not snake.alive:
                rewards.append(0)
                dones.append(True)
                continue

            new_snake_head, old_snake_tail = snake.step(action)
            other_snakes = copy(self.snakes)
            other_snakes.remove(snake)

            # If snake collides to the wall or if snake collides himself
            if (not (0 <= new_snake_head[0] < self.size[0]) or not (0 <= new_snake_head[1] < self.size[1])) \
                    or new_snake_head in snake.snake_body[1:]:
                rewards.append(self.cumulative_rewards[i] + self.REWARD['dead'])
                dones.append(True)
                snake.alive = False
                if self.is_competitive:
                    snake.body_to_fruit(self.world)
                else:
                    snake.free()
                # remove snake from player
                # self.snakes.remove(snake)
                # add to available positions
                self.available_positions = self.available_positions | set(snake.snake_body)

            # If snake collides with other snakes
            elif any(new_snake_head in s.snake_body for s in other_snakes):
                rewards.append(self.cumulative_rewards[i] + self.REWARD['dead'])
                dones.append(True)
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
                rewards.append(self.cumulative_rewards[i] + self.REWARD['eat'])
                self.cumulative_rewards = [sum(x) for x in zip(self.cumulative_rewards, rewards)]
                dones.append(False)
            else:
                snake.hunger += 1
                rewards.append(self.cumulative_rewards[i] + self.REWARD['move'])
                dones.append(False)

        return rewards, dones
        #  TODO: If snake collides with other snakes but action is 'cutting' (5)
