import gym
import sys
import gym_snake
from time import sleep
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='snake-multiple-v0',
  help="""\
  Select environment ID.
""")
FLAGS, unparsed = parser.parse_known_args()

env = gym.make(FLAGS.env)
#env = gym.wrappers.Monitor(env, 'tmp_video')

for e in range(500):
    obs = env.reset()
    print(env.n_snakes)
    dones = [False for _ in range(env.n_snakes)]
    print(dones)
    while not all(dones):
        actions = [env.action_space.sample() for _ in range(env.n_snakes)]
        obs, rewards, dones, info = env.step(actions)
        print('rewards: {} dones: {}'.format(rewards, dones))
        env.render(mode='human')
        sleep(0.1)

    sleep(2)

print("Observation:", obs.shape)
env.renderer.close()
