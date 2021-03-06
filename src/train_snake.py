import os
import sys
import gym
import argparse
import sys
sys.path.append('../')
from baselines import logger
import gym_snake

import multiprocessing
import tensorflow as tf

import numpy as np
import ppo_multi_agent_new
from policies import CnnPolicy

import utils
from config import Config

parser = argparse.ArgumentParser(description='Train snakes to play Slitherin')
parser.add_argument("--n-snakes", type=int, default=1)
parser.add_argument("--expr-name", help="Experiment name")

def main():
    args = parser.parse_args()
    num_snakes = args.n_snakes
    Config.set_num_snakes(num_snakes)
    Config.set_directory(args.expr_name)  # experiment setting name : e.g. ppo_2_10x10
    logger.configure()

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    env = utils.make_basic_env('snake-new-multiple-v0', ncpu, 0, False)
    print("env space is ", env.observation_space)
    num_timesteps = 1e7 if num_snakes == 1 else 8e7

    ppo_multi_agent_new.learn(policy=CnnPolicy, env=env, nsteps=64, nminibatches=8,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1, save_interval=50,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=num_timesteps)

if __name__ == '__main__':
    main()
