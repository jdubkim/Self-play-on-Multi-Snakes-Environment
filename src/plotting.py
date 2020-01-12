import random
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

ppo_data1 = {
    'nsteps': [],
    'eprewmean_100': [],
    'nupdates': [],
    'value_loss': [],
    'time_elapsed': [],
    'policy_entropy': [],
    'explained_variance': [],
    'eplenmean': [],
    'clipfrac': [],
    'serial_timesteps': [],
    'total_timesteps': [],
    'policy_loss': [],
    'next_highscore': [],
    'approxkl': []
}

ppo_data2 = {
    'nsteps': [],
    'eprewmean_100': [],
    'nupdates': [],
    'value_loss': [],
    'time_elapsed': [],
    'policy_entropy': [],
    'explained_variance': [],
    'eplenmean': [],
    'clipfrac': [],
    'serial_timesteps': [],
    'total_timesteps': [],
    'policy_loss': [],
    'next_highscore': [],
    'approxkl': []
}

def read_file1(filename):
    data = pd.read_csv(filename)
    print(list(data.columns.values))
    # ppo_data['next_highscore'] = data['next_highscore']
    ppo_data1['eprewmean_100'] = data['eprewmean 100']
    ppo_data1['total_timesteps'] = data['total_timesteps']

def read_file2(filename):
    data = pd.read_csv(filename)
    # ppo_data['next_highscore'] = data['next_highscore']
    ppo_data2['eprewmean_100'] = data['eprewmean 100']
    ppo_data2['total_timesteps'] = data['total_timesteps']

def comparison(filename1, filename2):
    read_file1(filename1)
    ppo_data1['total_timesteps'] = ppo_data1['total_timesteps']
    ppo_data1['eprewmean_100'] = ppo_data1['eprewmean_100']
    self_play = plt.plot(ppo_data1['total_timesteps'][:9765], ppo_data1['eprewmean_100'][:9765], label='self play')
    read_file2(filename2)
    vanilla_ppo = plt.plot(ppo_data2['total_timesteps'], ppo_data2['eprewmean_100'], label='vanilla PPO', color='orange')
    # plt.legend([self_play, vanilla_ppo], ['self play', 'vanilla PPO'])
    plt.xlabel('steps')
    plt.ylabel('100 ep mean reward')
    plt.title('Vanilla PPO')
    axes = plt.gca()
    axes.set_ylim([-2,27])
    plt.gca().legend(('Self Play', 'Vanilla PPO'))
    plt.show()

def plot(filename):
    read_file1(filename)
    vanilla_ppo = plt.plot(ppo_data1['total_timesteps'], ppo_data1['eprewmean_100'], label='vanilla PPO')
    # plt.legend([self_play, vanilla_ppo], ['self play', 'vanilla PPO'])
    plt.xlabel('steps')
    plt.ylabel('100 ep mean reward')
    plt.title('2 Agents Adversarial Environment')
    axes = plt.gca()
    #axes.set_ylim([-2, 27])
    # axes.set_ylim([-2, 27])
    # plt.gca().legend(('Self Play', 'Vanilla PPO'))
    plt.show()

# comparison('ppo2_19x19.csv', 'single_ppo_19x19.csv')
plot('ppo2_10x10_adv.csv')