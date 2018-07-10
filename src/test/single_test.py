import gym
import sys
import gym_snake
from time import sleep
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='snake-single-v0',
  help="""\
  Select environment ID.
""")
FLAGS, unparsed = parser.parse_known_args()

env = gym.make(FLAGS.env)
#env = gym.wrappers.Monitor(env, 'tmp_video')

for e in range(500):
    obs = env.reset()
    done = False
    r = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        r += reward
        env.render(mode='human')
        sleep(0.01)

print("Observation:", obs.shape)
env.renderer.close()
