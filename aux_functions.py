#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:15:02 2019

@author: marius
"""

import numpy as np

# Initialize random user and base station locations within given limits
# And other parameters
def init_scenario(num_users, xlim, ylim, epsilon_distance, noise_power,
                  p_valid, eta):
    # Initialize user powers
    # Blind as they enter the cell
    user_power_idx = np.random.randint(0, len(p_valid), num_users) #(np.floor(len(p_valid)/2) * np.ones((num_users,))).astype(np.int)
    
    # Initialized reward prediction to 0
    reward_prediction = np.zeros((num_users,))
    
    # Num times user selected
    user_selection_count = np.zeros((num_users,))
    
    # Create and return scenario dictionary
    scenario = {'xlim': xlim,
                'ylim': ylim,
                'epsilon_distance': epsilon_distance,
                'num_users': num_users,
                'user_power_idx': user_power_idx,
                'noise_power': noise_power,
                'p_valid': p_valid,
                'eta': eta,
                'reward_prediction': reward_prediction,
                'user_selection_count': user_selection_count}
    return scenario

def set_locations(scenario, max_user_box):
    xlim = scenario['xlim']
    ylim = scenario['ylim']
    num_users = scenario['num_users']
    
    bs_location = np.asarray([np.random.randint(low=np.max(xlim)//2-2, high=np.max(xlim)//2+2, size=(1,)),
                          np.random.randint(low=np.max(ylim)//2-2, high=np.max(ylim)//2+2, size=(1,))]).T
    
    # Generate box sizes of random dimensions around each user
    xbox = np.random.randint(low=0, high=max_user_box+1, size=(num_users,))
    ybox = np.random.randint(low=0, high=max_user_box+1, size=(num_users,))

    # Place users at random starting locations
    user_locations = np.empty((num_users, 2))
    xlim_user = np.empty((num_users, 2))
    ylim_user = np.empty((num_users, 2))
    
    for user_idx in range(num_users):
        # Generate user locations
        user_locations[user_idx] = np.asarray([np.random.randint(low=np.min(xlim)+xbox[user_idx], high=np.max(xlim)-xbox[user_idx]),
                                               np.random.randint(low=np.min(ylim)+ybox[user_idx], high=np.max(ylim)-ybox[user_idx])])
        # Compute allowable movement limits (for restricted movement scenario)
        xlim_user[user_idx] = np.asarray([user_locations[user_idx][0]-xbox[user_idx], user_locations[user_idx][0]+xbox[user_idx]])
        ylim_user[user_idx] = np.asarray([user_locations[user_idx][1]-ybox[user_idx], user_locations[user_idx][1]+ybox[user_idx]])    
    
    scenario['bs_location'] = bs_location
    scenario['user_locations'] = user_locations
    scenario['xlim_user'] = xlim_user
    scenario['ylim_user'] = ylim_user
    
    return scenario

# Move all users in a random direction inside their own boxes
def move_random_boxed(scenario):
    # Original locations
    locations = scenario['user_locations']
    
    # Edges of box (for each user)
    xlim = scenario['xlim_user']
    ylim = scenario['ylim_user']
    
    # All possible movement operators
    movement_ops = np.asarray([-1, 0, 1])
    
    # For each user
    for user_idx in range(locations.shape[0]):
        # Generate possible new coordinates
        new_x = locations[user_idx][0] + movement_ops
        new_y = locations[user_idx][1] + movement_ops
        # Filter invalid ones
        new_x = new_x[(new_x >= xlim[user_idx][0]) & (new_x <= xlim[user_idx][1])]
        new_y = new_y[(new_y >= ylim[user_idx][0]) & (new_y <= ylim[user_idx][1])]
        # And pick a random one
        new_x = new_x[np.random.choice(len(new_x))]
        new_y = new_y[np.random.choice(len(new_y))]
        # Save new location
        locations[user_idx] = np.asarray([new_x, new_y])
    
    # Write new locations
    scenario['user_locations'] = locations
    
    return scenario

# Move all users in a random direction (diagonal included)
def move_random(scenario):
    # Original locations
    locations = scenario['user_locations']
    
    # Edges of box
    xlim = scenario['xlim']
    ylim = scenario['ylim']
    
    # All possible movement operators
    movement_ops = np.asarray([-1, 0, 1])
    
    # For each user
    for user_idx in range(locations.shape[0]):
        # Generate possible new coordinates
        new_x = locations[user_idx][0] + movement_ops
        new_y = locations[user_idx][1] + movement_ops
        # Filter invalid ones
        new_x = new_x[(new_x >= xlim[0]) & (new_x <= xlim[1])]
        new_y = new_y[(new_y >= ylim[0]) & (new_y <= ylim[1])]
        # And pick a random one
        new_x = new_x[np.random.choice(len(new_x))]
        new_y = new_y[np.random.choice(len(new_y))]
        # Save new location
        locations[user_idx] = np.asarray([new_x, new_y])
    
    # Write new locations
    scenario['user_locations'] = locations
    
    # Return new dictionary
    return scenario

# Genie-aided SINR computation for all users based on their grid positions
def update_sinr(scenario):
    # Transmit powers
    # Distances to base station
    user_bs_dist = np.sqrt(np.sum(np.square(scenario['user_locations'] - scenario[
            'bs_location']), axis=-1))
    # Replace zero distances with epsilon stabilizer
    user_bs_dist[user_bs_dist == 0] = scenario['epsilon_distance']
    
    # User signal power
    user_signal = scenario['p_valid'][scenario['user_power_idx']] * np.square(1/user_bs_dist)
    # Interference sums between users
    user_interference = np.dot(np.square(1/user_bs_dist), scenario['p_valid'][scenario['user_power_idx']]) - user_signal
    
    # Update SINR values
    user_sinr = user_signal / (user_interference + scenario['noise_power'])
    
    # Write and return
    scenario['user_sinr'] = user_sinr
    return scenario

def pick_user(scenario, bandit_strategy, epsilon, time_idx, c = None):    
    # Choose user
    if bandit_strategy == 'random':
        user_idx = np.random.randint(low=0, high=scenario['num_users'])
        
    elif bandit_strategy == 'ucb_exp' or bandit_strategy == 'ucb_avg':
        values = scenario['reward_prediction'] + c * np.sqrt((time_idx + 1) / (scenario['user_selection_count']+10e-10))
        user_idx = np.random.choice(np.where(values == np.max(values))[0])
        
    elif bandit_strategy == 'avg' or bandit_strategy == 'exp':
        # Epsilon greedy
        if np.random.binomial(1 , 1 - epsilon, 1):   
            values = scenario['reward_prediction']
            user_idx = np.random.choice(np.where(values == np.max(values))[0])
        else:
            user_idx = np.random.randint(low=0, high=scenario['num_users'])    
    else:
        # Error
        return -1

#    if bandit_strategy != 'random':
#        if scenario['user_power_idx'][user_idx] == 0 or scenario['user_power_idx'][user_idx] == (len(scenario['p_valid']) - 1):
#            user_idx = np.random.choice(np.setdiff1d(np.arange(scenario['num_users']), user_idx))
    
    # Update user selection count
    scenario['user_selection_count'][user_idx] += 1
    
    return user_idx

def update_estimate(scenario, reward, bandit_strategy, time_idx, alpha = None):
    if bandit_strategy == 'avg' or bandit_strategy == 'ucb_avg':
        scenario['reward_prediction'] += (reward - scenario['reward_prediction'])/(time_idx + 1)
    elif bandit_strategy == 'exp' or bandit_strategy == 'ucb_exp':
        scenario['reward_prediction'] += alpha*(reward - scenario['reward_prediction'])
        
    return scenario
    
# One-step power adjustment scheme
def user_power_adjust(scenario, user_update_idx, algorithm, step_size):
    cur_pow_idx = scenario['user_power_idx'][user_update_idx]
    cur_pow = scenario['p_valid'][cur_pow_idx]
    cur_sinr = scenario['user_sinr'][user_update_idx]
    min_sinr = scenario['eta'][user_update_idx]
    
    # Step Update
    if algorithm == 'step':
        if cur_sinr >= min_sinr:
            new_pow = scenario['p_valid'][max(cur_pow_idx - 1, 0)]
            new_sinr = new_pow * (cur_sinr / cur_pow)
            
            if new_sinr >= min_sinr:
                scenario['user_power_idx'][user_update_idx] = max(cur_pow_idx - 1, 0)
                
        else:
            scenario['user_power_idx'][user_update_idx] = min(cur_pow_idx + 1, len(scenario['p_valid'])-1)
    elif algorithm == 'direct':
        if step_size:
            if cur_sinr > min_sinr:
                cur_sinr = min_sinr * step_size
            elif cur_sinr <= min_sinr:
                cur_sinr = min_sinr / step_size
        
        new_pow = min_sinr * cur_pow / cur_sinr
        
        scenario['user_power_idx'][user_update_idx] = np.argmin(np.square(new_pow - scenario['p_valid']))

    return scenario

def reset_users(scenario):
    scenario['user_power_idx'] = np.random.randint(0, len(scenario['p_valid']), scenario['num_users'])
    
    return scenario

# Capacity reward with a negative penalty when QoS is violated
def reward(scenario, reward_function):
    # Compute individual capacity
    actual_reward = scenario['user_sinr'] > scenario['eta']
    
    if reward_function == 'actual':
        learning_reward = actual_reward
    elif reward_function == 'delta':
        learning_reward = np.abs(np.log2(1 + scenario['user_sinr']) - np.log2(1+ scenario['eta']))    
    
    
    return learning_reward, actual_reward