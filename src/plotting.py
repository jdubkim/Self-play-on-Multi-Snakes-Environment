import random
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

ppo_data = {
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

def read_file(filename):
    data = pd.read_csv(filename)
    # ppo_data['next_highscore'] = data['next_highscore']
    ppo_data['eprewmean_100'] = data['explained_variance']
    ppo_data['total_timesteps'] = data['total_timesteps']

read_file('log_July_26_2018.csv')
plt.plot(ppo_data['total_timesteps'], ppo_data['eprewmean_100'])
plt.xlabel('steps')
plt.ylabel('100 ep mean reward')
plt.title('Single PPO reward')
plt.show()