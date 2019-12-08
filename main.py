#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:18:32 2019

@author: marius
"""
import copy
import numpy as np

from tqdm import tqdm

from aux_functions import *
#from aux_functions import init_scenario
#from aux_functions import move_random, update_sinr
#from aux_functions import user_power_adjust
#from aux_functions import capacity_reward


from matplotlib import pyplot as plt

## Control global seed
global_seed = 2019
np.random.seed(global_seed)

# Num iters
num_iters = 500

## System parameters
# Box limits
xlim = np.asarray([0, 60])
ylim = np.asarray([0, 60])

# Epsilon distance stabilizer
# If user and bs have the same coordinates,
# replace their pairwise distance with this
epsilon_distance = 1e-2

# Number of users
num_users = 4

# Valid power levels
p_steps = 200
p_valid = np.linspace(0, 1000, p_steps)

# Noise power (universal)
noise_power = 10 ** (-23 / 10)

# User SINR (QOS) requirements
eta = 10 ** (-5.7 * np.ones((num_users,)) / 10)

# Reward penalty for not satisfying QoS
eta_penalty = 5

# Number of timesteps (can also be infinite)
num_steps = 1000

# Bandit strategy
# Options: avg, exp, ucb_avg, ucb_exp, random
bandit_strategy = 'exp'
c = 1 #required for ucb
c_increment = 0
alpha = 0. #required for 
epsilon = 0.

# User power update algorithm
algorithm = 'direct'

avg_learning_reward_hist = np.zeros((num_steps,num_users))
avg_actual_reward_hist = np.zeros((num_steps,num_users))
avg_sinr_hist = np.zeros((num_steps,num_users))
avg_user_power_hist = np.zeros((num_steps,num_users))

actual_reward_iter = []
user_power_iter = []
user_dist_iter = []

for iter_num in tqdm(range(num_iters)):
    np.random.seed(global_seed+iter_num)
    
    ## Scenario dictionary
    scenario = init_scenario(num_users, xlim, ylim, epsilon_distance,
                             noise_power, p_valid, eta,
                             eta_penalty)
    
    ## Time evolution
    learning_reward_hist = []
    actual_reward_hist = []
    sinr_hist = []
    user_power_hist = []
    user_idx_hist = []
    
    # For each timestep
    for time_idx in range(num_steps):
        # Increment user locations
        #if np.mod(time_idx+1, 40) == 0:
            #c += c_increment
            #scenario = move_random(scenario)
        
        # Update SINR values
        scenario = update_sinr(scenario)
        
        sinr = scenario['user_sinr']
        sinr_hist.append(scenario['user_sinr'])
        
        # Pick random user
        #user_idx = np.random.randint(low=0, high=num_users)
        
        # Pick user: avg, exp, ucb, random
        user_idx = pick_user(scenario, bandit_strategy, epsilon, time_idx, c)
        
        user_idx_hist.append(user_idx)
        
        # Send SINR update to a specific user
        scenario = user_power_adjust(scenario, user_idx, algorithm, step_size=1.05)
        
        user_power = scenario['p_valid'][scenario['user_power_idx']]
        
        user_power_hist.append(user_power)
        
        # Receive new sum-rate as reward
        learning_reward, actual_reward = capacity_reward(scenario)
        
        # Store metrics
        learning_reward_hist.append(learning_reward)
        
        actual_reward_hist.append(actual_reward)
        
        # Update action-value estimate
        scenario = update_estimate(scenario, learning_reward, bandit_strategy, time_idx, alpha)
    
    # Convert to arrays
    avg_learning_reward_hist += np.asarray(learning_reward_hist)
    avg_actual_reward_hist += np.asarray(actual_reward_hist)
    avg_sinr_hist += np.asarray(sinr_hist)
    avg_user_power_hist += np.asarray(user_power_hist)
    user_idx_hist += np.asarray(user_idx_hist)

    actual_reward_iter.append(actual_reward)
    user_power_iter.append(user_power)
    user_dist_iter.append(np.sqrt(np.sum(np.square(scenario['user_locations'] - scenario[
            'bs_location']), axis=-1)))

actual_reward_iter = np.asarray(actual_reward_iter)
user_power_iter = np.asarray(user_power_iter)
user_dist_iter = np.asarray(user_dist_iter)

failed_reward = actual_reward_iter[np.sum(actual_reward_iter, axis = -1) <= 1]
failed_power = user_power_iter[np.sum(actual_reward_iter, axis = -1) <= 1]
failed_dist = user_dist_iter[np.sum(actual_reward_iter, axis = -1) <= 1]

# Compute total rewards
sum_learning_reward_hist = np.sum(avg_learning_reward_hist, axis=1) / num_iters
sum_actual_reward_hist = np.sum(avg_actual_reward_hist, axis=1) / num_iters

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12,6))
fig.suptitle('RL Performance', fontsize=16, y=1.02)
         
axes[0].plot(sum_actual_reward_hist)
axes[0].set_title('Actual Reward History', fontsize = 14)
axes[0].set_xlabel('Time', fontsize = 14)

axes[1].hist(user_power_iter.flatten(), bins=100)
axes[1].set_title('User Power', fontsize = 14)
axes[1].set_xlabel('Bins', fontsize = 14)

plt.tight_layout()
plt.show()