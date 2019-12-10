import pickle
import numpy as np
import datetime as datetime
from matplotlib import pyplot as plt

filename_random = 'results/20191209-231203_random_actual.pkl'
filename_exp_actual = 'results/20191209-231213_exp_actual.pkl'
filename_exp_delta = 'results/20191209-231226_exp_delta.pkl'

with open(filename_random, 'rb') as f:
    data = pickle.load(f)
    reward_random = data['actual_reward'] / data['initialized_scenario']['num_users']
    
with open(filename_exp_actual, 'rb') as f:
    data = pickle.load(f)
    reward_exp_actual = data['actual_reward'] / data['initialized_scenario']['num_users']
    
with open(filename_exp_delta, 'rb') as f:
    data = pickle.load(f)
    reward_exp_delta = data['actual_reward'] / data['initialized_scenario']['num_users']
    
# Plot
fig, axes = plt.subplots(1, 1, figsize=(8,6))
         
axes.plot(reward_random, '-k', linewidth=3, label='Random Update')
axes.plot(reward_exp_actual, '-b', linewidth=3, label='Bandit with Binary Reward, \u03B1 = 0.5, \u03B5 = 0.01')
axes.plot(reward_exp_delta, '-r', linewidth=3, label='Bandit with Soft Reward, \u03B1 = 0.5, \u03B5 = 0.01')
axes.set_title('Reward History', fontsize = 16)
axes.set_xlabel('Time', fontsize = 16)
axes.set_ylabel('% of Successful Users', fontsize = 16)
axes.legend(fontsize=12)

axes.set_xticks(np.arange(0, 1001, 100))
axes.set_yticks(np.arange(.2, 1.01, 0.1))

for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 

for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 

plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('comparison.eps', format='eps', bbox_inches='tight')

