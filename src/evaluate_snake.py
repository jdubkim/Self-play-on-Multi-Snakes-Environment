import gym
import numpy as np
from policies import CnnPolicy
import time
import joblib
import multiprocessing
import tensorflow as tf
import sys
import argparse
import gym_snake

import envs
from config import Config
import utils
from utils import WarpFrame

parser = argparse.ArgumentParser(description='Visualize a trained snakes in the Slitherin environment')
parser.add_argument("--n-snakes", default=1, help="number of snakes in the environment")
parser.add_argument("--file", type=str, default="example_model.pkl", help="file to load")

def load_act_model(load_file, model_scope, env, nenvs=1, num_actions=5):
    print('Loading from...', load_file)

    ob_shape = utils.get_shape(env.observation_space)
    ac_space = env.action_space

    sess = tf.get_default_session()

    act = CnnPolicy(sess, ob_shape, ac_space, nenvs, 1, model_scope, reuse=False)

    with tf.variable_scope(model_scope):
        params = tf.trainable_variables(model_scope)

    loaded_params = joblib.load(Config.MODEL_DIR + load_file)
    restores = []
    for p, loaded_p in zip(params, loaded_params):
        restores.append(p.assign(loaded_p))
    sess.run(restores)

    return act

def main():
    args = parser.parse_args()
    num_snakes = int(args.n_snakes)
    Config.set_num_snakes(num_snakes)
    agent_file = args.file

    tf.Session().__enter__()

    env = gym.make('snake-adversarial-v0')
    env = WarpFrame(env)

    act0 = load_act_model(agent_file, 'model', env)
    act1 = load_act_model(agent_file, 'model_2', env)
    act2 = None
    if num_snakes == 3:
        act2 = load_act_model(agent_file, 'model_3', env)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        t_step = 0

        env.render()

        has_transitioned = False
        last_done = False

        states0s = act0.initial_state
        states1s = act1.initial_state
        states2s = None
        if act2 is not None:
            states2s = act2.initial_state
        
        while True:
            obs = obs.__array__()
            obs = obs.reshape((1,) + np.shape(obs))

            obs0 = obs[:,:,:,0:3]
            obs1 = obs[:,:,:,3:6]
            obs2 = None
            if num_snakes == 3:
                obs2 = obs[:,:,:,6:9]

            action0, _, states0s, _ = act0.step(obs0, states0s, [last_done])
            action1, _, states1s, _ = act1.step(obs1, states1s, [last_done])
            if num_snakes == 3:
                action2, _, states2s, _ = act2.step(obs2, states2s, [last_done])

            action = [action0[0], action1[0]]
            if num_snakes == 3:
                action.append(action2[0])

            obs, rew, done, info = env.step(action)
            last_done = done
            episode_rew += rew

            sleep_time = 0

            if info["num_snakes"] <= 1:
                sleep_time = .1

                if not has_transitioned:
                    has_transitioned = True
            else:
                sleep_time = .1

            env.render()

            if sleep_time > 0:
                time.sleep(sleep_time)

            t_step += 1

            if info["num_snakes"] <= 0:
                break

    return episode_rew, ep_len

if __name__ == '__main__':
    main()
