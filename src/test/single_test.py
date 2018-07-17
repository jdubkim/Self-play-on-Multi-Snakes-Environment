import gym
import itertools
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import sys
sys.path.append('../')
import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

import gym_snake

BATCH_SIZE = 32

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='snake-single-v0')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-path', type=str, default=None)

    args = parser.parse_args()

    # make_session first argument : num of cpus
    with U.make_session(8):
        env = gym.make(args.env)
        print("observation space is ", env.observation_space)
        print("action space is ", env.action_space)
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 5, 1), (64, 5, 1)],
            hiddens=[256],
            dueling=bool(args.dueling)
        )

        act = deepq.learn(env,
                          q_func=model,
                          lr=1e-4,
                          max_timesteps=2000000,
                          buffer_size=50000,
                          train_freq=100,
                          exploration_fraction=0.1,
                          exploration_final_eps=0.02,
                          gamma=0.99,
                          print_freq=10,
                          checkpoint_freq=args.checkpoint_freq,
                          checkpoint_path=args.checkpoint_path,
                          param_noise=True)
        act.save("../models/single-dqn/single_dqn_model_final.pkl")
        env.close()

if __name__ == '__main__':
    main()

# import gym
# import sys
# import gym_snake
# sys.path.append('../model/')
# import dqn2015
# from time import sleep
# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default='snake-single-v0',
#   help="""\
#   Select environment ID.
# """)
# FLAGS, unparsed = parser.parse_known_args()
#
# env = gym.make(FLAGS.env)
# env.reset()
# qNet = dqn2015.DQN2015(env)
# qNet.run()
# #env = gym.wrappers.Monitor(env, 'tmp_video')
#
# # for e in range(500):
# #     obs = env.reset()
# #     done = False
# #     r = 0
# #     while not done:
# #         action = env.action_space.sample()
# #         obs, reward, done, info = env.step(action)
# #         r += reward
# #         env.render(mode='human')
# #         sleep(0.01)
#
# env.close()
