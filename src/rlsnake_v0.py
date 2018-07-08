import gym
import sys
sys.path.append('../gym_snake')
import gym_snake
import config
import setting
import dqn2015

env = gym.make('snake-v0')
env.__init__(num_agents=1)
#env = gym.wrappers.Monitor(env, directory="gym-results/", force=True)
env.reset()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
#
# qNet = dqn2015.DQN2015(env)
# qNet.run()
