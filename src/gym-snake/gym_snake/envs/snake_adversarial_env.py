import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
from config import Config

class SnakeAdversarial(gym.Env):
