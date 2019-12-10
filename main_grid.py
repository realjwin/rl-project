import pickle
import numpy as np
import datetime as datetime
from matplotlib import pyplot as plt
import tqdm as tqdm

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
alpha = np.array([.1, .2, .5, .9, .99])
epsilon = np.array([0, .01, .05, .1])
c = None
step_size = None #dampened update (delta)

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

# Saved variables
actual_reward_list = []
learning_reward_list = []
user_power_hist_list = []
sinr_hist_list = []

# Initialize scenario
initialized_scenario = init_scenario(num_users, xlim, ylim, epsilon_distance, noise_power, p_valid, eta)

for epsilon_idx, epsilon_val in enumerate(epsilon):
    for alpha_idx, alpha_val in enumerate(alpha):
        print('epsilon: {}, alpha: {}'.format(epsilon_val, alpha_val))
        
        actual_reward, learning_reward, user_power_hist, sinr_hist = run_bandit(
                    num_iterations, num_steps, initialized_scenario, power_adjust,
                    reward_function, bandit_strategy, alpha_val, epsilon_val, step_size,
                    move_strategy, move_steps, move_stepsize, c)
        
        actual_reward_list.append(actual_reward)
        learning_reward_list.append(learning_reward)
        user_power_hist_list.append(user_power_hist)
        sinr_hist_list.append(sinr_hist)

ts = datetime.datetime.now()

filename = ts.strftime('%Y%m%d-%H%M%S') + '_' + bandit_strategy + '_' + reward_function + '_grid.pkl'
filepath = 'results/' + filename

with open(filepath, 'wb') as f:
    save_dict = {
            'actual_reward_list': actual_reward_list,
            'learning_reward_list': learning_reward_list,
            'user_power_hist_list': user_power_hist_list,
            'sinr_hist_list': sinr_hist_list,
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