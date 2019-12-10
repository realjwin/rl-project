import pickle
import numpy as np
import datetime as datetime
from matplotlib import pyplot as plt

filename_random_movement = 'results/20191209-214608_random_delta_grid_movement.pkl'

filename_bandit_movement = 'results/20191209-215502_exp_delta_grid_movement.pkl'

with open(filename_random_movement, 'rb') as f:
    data = pickle.load(f)
    reward_list_random = data['actual_reward_list']
    move_stepsize = data['move_stepsize']
    num_users = data['initialized_scenario']['num_users']

with open(filename_bandit_movement, 'rb') as f:
    data = pickle.load(f)
    reward_list_bandit = data['actual_reward_list']
    
# Plot
fig, axes = plt.subplots(1, 1, figsize=(8,6))

for move_stepsize_idx, move_stepsize_val in enumerate(move_stepsize): 
    axes.plot(reward_list_random[move_stepsize_idx] / num_users, linewidth=3, label='Random, Step={}'.format(move_stepsize_val))
    axes.plot(reward_list_bandit[move_stepsize_idx] / num_users, linewidth=3, label='Bandit, Step={}'.format(move_stepsize_val))

axes.set_title('Reward History', fontsize = 16)
axes.set_xlabel('Time', fontsize = 16)
axes.set_ylabel('% of Successful Users', fontsize = 16)
axes.legend(fontsize=12)

axes.set_xlim(0, 3000)
axes.set_ylim(.1, 1)
axes.set_xticks(np.arange(0, 3001, 500))
axes.set_yticks(np.arange(.1, 1.01, 0.1))

for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
    #tick.label.set_rotation('vertical')
    
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
    #tick.label.set_rotation('vertical')

plt.grid(linestyle='--')
plt.tight_layout()
plt.savefig('comparison.eps', format='eps', bbox_inches='tight')