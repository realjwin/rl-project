import pickle
import numpy as np
import datetime as datetime
from matplotlib import pyplot as plt

from aux_functions import *
from bandit import run_bandit

# Seed if desired
#global_seed = 2019
#np.random.seed(global_seed)

#--- SIMULATION VARIABLES ---#
num_iterations = 10
num_steps = 1000

# Bandit strategy
# Options: avg, exp, ucb_avg, ucb_exp, random
bandit_strategy = 'exp'
alpha = 0.5
epsilon = 0.
c = None
step_size = 1.0725 #dampened update (delta)

# Reward function
# Options: actual or delta (binary or soft)
reward_function = 'delta'

#--- SCENARIO VARIABLES ---#

# Number of users
num_users = 4

# Movement strategy
# Options: stationary, random
move_strategy = 'stationary'
move_steps = None #how many steps until movement
move_stepsize = None #how much it moves in one step

# User power update algorithm
# Options: direct, step
power_adjust='direct'

# Power levels
p_steps = 200
p_valid = np.linspace(1e-4, 1000, p_steps)

# User SINR (QOS) requirements
eta = 10 ** (-5.7 * np.ones((num_users,)) / 10)

# Noise power (universal)
noise_power = 10 ** (-23 / 10)

# Box size
box_limits = 6000 #centimeters

# Miscellaneous
epsilon_distance = 1e-2 #in case user shares coordinates w/ bs
xlim = np.asarray([0, box_limits])
ylim = np.asarray([0, box_limits])

#--- RUN BANDIT ---#

# Initialize scenario
initialized_scenario = init_scenario(num_users, xlim, ylim, epsilon_distance, noise_power, p_valid, eta)

actual_reward, learning_reward, user_power_hist, sinr_hist = run_bandit(
            num_iterations, num_steps, initialized_scenario, power_adjust,
            reward_function, bandit_strategy, alpha, epsilon, step_size,
            move_strategy, move_steps, move_stepsize, c)

#--- PLOT ---#
fig, axes = plt.subplots(1, 1, figsize=(8,6))
fig.suptitle('RL Performance', fontsize=16, y=1.02)
         
axes.plot(actual_reward)
axes.set_title('Actual Reward History', fontsize = 14)
axes.set_xlabel('Time', fontsize = 14)

plt.tight_layout()
plt.show()

#max_loc = np.argmax(actual_reward)
#print('Average power at {}: {}, Average power at end: {}'.format(max_loc, np.mean(user_power_hist[:,max_loc,:]), np.mean(user_power_hist[:,-1,:])))

#--- SAVE ---#

ts = datetime.datetime.now()

filename = ts.strftime('%Y%m%d-%H%M%S') + '_' + bandit_strategy + '_' + reward_function + '.pkl'
filepath = 'results/' + filename

with open(filepath, 'wb') as f:
    save_dict = {
            'actual_reward': actual_reward,
            'learning_reward': learning_reward,
            'user_power_hist': user_power_hist,
            'sinr_hist': sinr_hist,
            'num_iterations': num_iterations,
            'num_steps': num_steps,
            'initialized_scenario': initialized_scenario,
            'alpha': alpha,
            'epsilon': epsilon,
            'step_size': step_size,
            'move_strategy': move_strategy,
            'move_steps': move_steps,
            'move_stepsize': move_stepsize}
    
    pickle.dump(save_dict, f)