import gym
from time import sleep
import gym_snake
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='snake-single-v0',
  help="""\
  Select environment ID.
""")
FLAGS, unparsed = parser.parse_known_args()
env = gym.make(FLAGS.env)

for _ in range(10):
    obs = env.reset()
    done = False
    r = 0
    print('init_state: {} example action: {}'.format(obs, env.action_space.sample()))
    while not done:
        env.render()
        obs, reward, done, info = env.step(eval(input('')))
        print('state: {} reward: {} done: {}'.format(obs, reward, done))
        sleep(0.01)