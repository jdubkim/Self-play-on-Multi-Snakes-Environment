import gym
from time import sleep
import numpy as np
import gym_snake
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='snake-single-v0',
  help="""\
  Select environment ID.
""")
FLAGS, unparsed = parser.parse_known_args()
env = gym.make(FLAGS.env)

def keyboard_input():
    inpt = input()
    if inpt == "h" or inpt == "H":
        return 3
    elif inpt == "j" or inpt == "J":
        return 2
    elif inpt == "k" or inpt == "K":
        return 0
    else:
        return 1

env.reset()
for _ in range(10):
    obs = env.reset()
    done = False
    r = 0
    print('example action: {}'.format(env.action_space.sample()))
    while not done:
        env.render(mode='human')
        action = keyboard_input()
        if action != 0:
            print("good")
        obs, reward, done, info = env.step(action)
        print('reward: {} done: {}'.format(reward, done))
        sleep(0.01)

