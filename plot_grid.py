import pickle
import numpy as np
import datetime as datetime
from matplotlib import pyplot as plt



filename = 'results/20191209-162550_exp_delta_grid.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f)
    reward_list = data['actual_reward_list']
    epsilon = data['epsilon']
    alpha = data['alpha']
    num_users = data['initialized_scenario']['num_users']

# Plot
fig, axes = plt.subplots(1, 1, figsize=(8,6))

for epsilon_idx, epsilon_val in enumerate(epsilon):
#for alpha_idx, alpha_val in enumerate(alpha):
    alpha_idx = 2
    alpha_val = alpha[alpha_idx]
    #epsilon_idx = len(epsilon) - 1
    #epsilon_val = epsilon[epsilon_idx]
    
    reward_idx = epsilon_idx*len(alpha) + alpha_idx  
    axes.plot(reward_list[reward_idx] / num_users, label='\u03B5 = {}, \u03B1 = {}'.format(epsilon_val, alpha_val))

axes.set_title('Reward History', fontsize = 16)
axes.set_xlabel('Time', fontsize = 16)
axes.set_ylabel('% of Successful Users', fontsize = 16)
axes.legend(fontsize=10)

axes.set_xlim(0, 1000)
axes.set_ylim(.4, 1)
axes.set_xticks(np.arange(0, 1001, 100))
axes.set_yticks(np.arange(.4, 1.01, 0.1))

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

for epsilon_idx, epsilon_val in enumerate(epsilon):
    for alpha_idx, alpha_val in enumerate(alpha):
        reward_idx = epsilon_idx*len(alpha) + alpha_idx  
        
        val = reward_list[reward_idx][-1]
        
        if val > val_max:
            max_alpha = alpha_idx
            max_epsilon = epsilon_idx
            val_max = val
            
print('epsilon: {}, alpha: {}'.format(epsilon[max_epsilon], alpha[max_alpha]))