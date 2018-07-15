import gym
import sys
import gym_snake
sys.path.append('../model/')
import dqn2015
from time import sleep
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='snake-single-v0',
  help="""\
  Select environment ID.
""")
FLAGS, unparsed = parser.parse_known_args()

env = gym.make(FLAGS.env)
env.reset()
qNet = dqn2015.DQN2015(env)
qNet.run()
#env = gym.wrappers.Monitor(env, 'tmp_video')

# for e in range(500):
#     obs = env.reset()
#     done = False
#     r = 0
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         r += reward
#         env.render(mode='human')
#         sleep(0.01)

env.renderer.close()
