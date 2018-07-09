import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import random
from collections import deque
from pathlib import Path
from typing import List

import dqn
import setting

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 5000

class DQN2015:

    def __init__(self, env: gym.Env):

        self.env = env # environment

        self.input_size = np.ndarray([env.observation_space.shape[0], env.observation_space.shape[1], 3]) # 224 * 256 * 3
        self.output_size = 6 # Num of Arrow Keys

    def replay_train(self, mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:

        x_stack = np.empty(0).reshape(0, mainDQN.input_size)
        y_stack = np.empty(0).reshape(0, mainDQN.output_size)

        # Get stored information from the buffer
        for state, action, reward, next_state, done in train_batch:
            if state is None:
                print("None State, ", action, " , ", reward, " , ", next_state, " , ", done)
            else:
                Q = mainDQN.predict(state)

                if done:
                    Q[0, action] = reward
                else:
                    Q[0, action] = reward + DISCOUNT_RATE * np.max(targetDQN.predict(next_state))

                y_stack = np.vstack([y_stack, Q])
                x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size)])  # to fit for super mario

        # Train our network using target and predicted Q values on each episode
        return mainDQN.update(x_stack, y_stack)

    def get_copy_var_ops(self, *, dest_scope_name="target", src_scope_name="main"):

        # Copy variables in mainDQN to targetDQN
        # Update weights in mainDQN to targetDQN
        op_holder = []

        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder

    def bot_play(self, mainDQN):
        # See our trained network in action
        state = self.env.reset()
        reward_sum = 0
        while True:
            self.env.render()
            action = np.argmax(mainDQN.predict(state))
            state, reward, done, _ = self.env.step(action)
            reward_sum += reward
            if done:
                print("Total score: {}".format(reward_sum))
                break

    def run(self):

        print(self.env.observation_space.shape[0], self.env.observation_space.shape[1], self.output_size)
        # store the previous observations in replay memory
        replay_buffer = deque()

        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, self.input_size, self.output_size, name="main")
            targetDQN = dqn.DQN(sess, self.input_size, self.output_size, name="target")
            self.saver = tf.train.Saver()

            model_stored = Path("/models/")
            if model_stored.is_file():
                self.saver.restore(sess, "/models/model.ckpt")
                print("Existing model restored")
            else:
                tf.global_variables_initializer().run()


            # initial copy q_net -> target_net
            copy_ops = self.get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
            sess.run(copy_ops)

            for episode in range(MAX_EPISODES):
                e = 1. / ((episode / 10) + 1)
                done = False
                max_distance = 0
                prev_pos = 0
                state = self.env.reset()

                while not done:
                    if np.random.rand(1) < e or state is None or state.size == 1:
                        action = self.env.action_space.sample()
                    else:
                        # action = np.argmax(mainDQN.predict(state))
                        action = mainDQN.predict(state).flatten().tolist()  #flatten it and change it as a list
                        for i in range(len(action)):  #the action list has to have only integer 1 or 0
                            if action[i] > 0.5:
                                action[i] = 1  #integer 1 only, no 1.0
                            else:
                                action[i] = 0  #integer 0 only, no 0.0

                    # Get new state and reward from environment
                    next_state, reward, done, info = self.env.step(action)
                    #(info)

                    current_distance = info['distance']
                    if done:  # Death or stayed more than 10000 steps
                        reward -= 3

                    if current_distance > prev_pos: # Move right
                        reward += 0.5
                    elif current_distance < prev_pos: # Move left
                        reward -= 1.0

                    if current_distance - prev_pos > 8: # Move right fast
                        reward += 1.0
                    elif current_distance - prev_pos < -8: # Move left fast
                        reward -= 1.5

                    #if is_level_cleared:
                    # reward += 2

                    prev_pos = current_distance

                    # Save the experience to our buffer
                    replay_buffer.append((state, action, reward, next_state, done))
                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()

                    state = next_state

                # Check if second level is unlocked
                if not info['locked_levels'][1]:
                    pass

                if episode % 10  == 1:  # train every 10 episode
                    # Get a random batch of experiences
                    for _ in range(50):
                        minibatch = random.sample(replay_buffer, 10)
                        loss, _ = self.replay_train(mainDQN, targetDQN, minibatch)

                    print("Loss: ", loss)
                    # copy q_net -> target_net
                    sess.run(copy_ops)
                    save_path = self.saver.save(sess, 'models/model.ckpt')
                    print("Model saved in file: %s" % save_path)

                print("Episode %d: Maximum distance: %d"%(episode, current_distance))

            # See our trained bot in action
            env2 = wrappers.Monitor(self.env, 'gym-results', force=True)

            for i in range(200):
                self.bot_play(mainDQN)

            env2.close()
