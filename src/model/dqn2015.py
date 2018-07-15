import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import random
from collections import deque
from pathlib import Path
from typing import List

import dqn

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 100000
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 1000
MAX_EPISODES = 2000000

class DQN2015:

    def __init__(self, env: gym.Env):

        self.env = env  # environment
        print("observation space is ", env.observation_space.shape)
        self.input_size = np.ndarray([env.observation_space.shape[0], env.observation_space.shape[1], 3])  #  24 X 24 X 3
        self.output_size = 4  # Num of Arrow Keys

    def replay_train(self, mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:

        x_stack = np.empty(0).reshape(0, mainDQN.input_size.size)
        y_stack = np.empty(0).reshape(0, mainDQN.output_size)

        # Get stored information from the buffer
        for state, action, reward, next_state, done in train_batch:
            if state is None:
                pass
            else:
                #print("State, ", action, " , ", reward)
                Q = mainDQN.predict(state)

                if done:
                    Q[0, action] = reward
                else:
                    Q[0, action] = reward + DISCOUNT_RATE * np.max(targetDQN.predict(next_state))

                y_stack = np.vstack([y_stack, Q])
                x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size.size)])

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

        # store the previous observations in replay memory
        replay_buffer = deque(maxlen=REPLAY_MEMORY)

        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, self.input_size, self.output_size, name="main")
            targetDQN = dqn.DQN(sess, self.input_size, self.output_size, name="target")
            #  merged= tf.summary.merge_all()
            #  writer = tf.summary.FileWriter("tensorboard/dqn/" + "/single", sess.graph)
            #  writer.add_graph(sess.graph)

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
            print("Ready to train")

            for episode in range(MAX_EPISODES):
                e = max(1. / ((episode / 20) + 1), 0.02)
                done = False
                step_count = 0
                state = self.env.reset()

                while not done:
                    self.env.render()
                    if np.random.rand() < e:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(mainDQN.predict(state))

                    # Get new state and reward from environment
                    next_state, reward, done, _ = self.env.step(action)
                    variable_summaries(reward)

                    # Save the experience to our buffer
                    replay_buffer.append((state, action, reward, next_state, done))

                    if len(replay_buffer) > BATCH_SIZE:
                        minibatch = random.sample(replay_buffer, BATCH_SIZE)
                        loss, _ = self.replay_train(mainDQN, targetDQN, minibatch)
                        #  print("Loss: ", loss)

                    if len(replay_buffer) > REPLAY_MEMORY:
                        replay_buffer.popleft()

                    # if episode % 10 == 0:
                    #     summary = sess.run(merged)
                    #     writer.add_summary(summary, episode)

                    if step_count % TARGET_UPDATE_FREQUENCY == 0:
                        sess.run(copy_ops)

                    state = next_state
                    step_count += 1

                if episode % 1000 == 0:  # save model every 1000 episode
                    save_path = self.saver.save(sess, 'models/model.cpkt', global_step=episode)
                    print("Model saved in file: %s" % save_path)

                print("Episode: {}  score: {}".format(episode, reward))

            self.env.close()
            #  writer.flush()
            #  writer.close()

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
