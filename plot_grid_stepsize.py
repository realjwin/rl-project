import pickle
import numpy as np
import datetime as datetime
from matplotlib import pyplot as plt



filename = 'results/20191209-195031_exp_delta_grid_stepsize.pkl' #20191209-183604_exp_delta_grid_stepsize.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f)
    reward_list = data['actual_reward_list']
    epsilon = data['epsilon']
    alpha = data['alpha']
    step_size = data['step_size']
    num_users = data['initialized_scenario']['num_users']

filename = 'results/20191209-174917_exp_delta_grid.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f)
    reward_list_other = data['actual_reward_list']
    epsilon = data['epsilon']
    alpha = data['alpha']
    
    no_step = reward_list_other[2]


# Plot
fig, axes = plt.subplots(1, 1, figsize=(8,6))

for step_size_idx, step_size_val in enumerate(step_size):    
    axes.plot(reward_list[step_size_idx] / num_users, linewidth=3, label='\u0394 = {}'.format(step_size_val))

axes.plot(no_step / num_users, linewidth=3, label='Direct Update')

axes.set_title('Reward History', fontsize = 16)
axes.set_xlabel('Time', fontsize = 16)
axes.set_ylabel('% of Successful Users', fontsize = 16)
axes.legend(fontsize=12)

axes.set_xlim(0, 1000)
axes.set_ylim(.1, 1)
axes.set_xticks(np.arange(0, 1001, 100))
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

val_max = 0

for step_size_idx, step_size_val in enumerate(step_size):        
        val = reward_list[step_size_idx][-1]
        
        if val > val_max:
            max_step_size = step_size_idx
            val_max = val
            
print('step_size: {}'.format(step_size[max_step_size]))