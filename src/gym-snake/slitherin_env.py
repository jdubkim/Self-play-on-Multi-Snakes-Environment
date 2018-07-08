import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from copy import deepcopy

from pygame.locals import *
import pygame
import time
from copy import deepcopy

class SlitherinEnv(gym.Env):
    AGENT_COLORS = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255)
    ]

    class Agent:
        def __init__(self, x, y, spacing, length=3, direction=0):
            self.init_length = length
            self.length = length

            self.direction = direction

