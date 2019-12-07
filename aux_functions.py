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
                  p_valid, eta, eta_penalty):
    # Place the base station near (or in) the center
    bs_location = np.asarray([np.random.randint(low=np.max(xlim)//2-2, high=np.max(xlim)//2+2, size=(1,)),
                              np.random.randint(low=np.max(ylim)//2-2, high=np.max(ylim)//2+2, size=(1,))]).T
    # Place users at random starting locations
    user_locations = np.asarray([np.random.randint(low=np.min(xlim), high=np.max(xlim), size=(num_users,)),
                                 np.random.randint(low=np.min(ylim), high=np.max(ylim), size=(num_users,))]).T
    # Initialize user powers
    # Blind as they enter the cell
    user_power_idx = (np.floor(len(p_valid)/2) * np.ones((num_users,))).astype(np.int)
    
    # Create and return scenario dictionary
    scenario = {'xlim': xlim,
                'ylim': ylim,
                'epsilon_distance': epsilon_distance,
                'num_users': num_users,
                'bs_location': bs_location,
                'user_locations': user_locations,
                'user_power_idx': user_power_idx,
                'noise_power': noise_power,
                'p_valid': p_valid,
                'eta': eta,
                'eta_penalty': eta_penalty}
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
    
# One-step power adjustment scheme
def user_power_adjust(scenario, user_update_idx):
    cur_pow_idx = scenario['user_power_idx'][user_update_idx]
    cur_pow = scenario['p_valid'][cur_pow_idx]
    cur_sinr = scenario['user_sinr'][user_update_idx]
    min_sinr = scenario['eta'][user_update_idx]
    
    # Perform direct update, step update limited.
    if cur_sinr >= min_sinr:
        new_pow = scenario['p_valid'][max(cur_pow_idx - 1, 0)]
        new_sinr = new_pow * (cur_sinr / cur_pow)
        
        if new_sinr >= min_sinr:
            scenario['user_power_idx'][user_update_idx] = max(cur_pow_idx - 1, 0)
            
    else:
        scenario['user_power_idx'][user_update_idx] = min(cur_pow_idx + 1, len(scenario['p_valid'])-1)
    
    #there is something wrong with this formula, changing to step update
    #new_power = scenario['eta'][user_update_idx] * scenario['user_powers'][user_update_idx] / scenario['user_sinr'][user_update_idx]
    #new_power = scenario['p_valid'][np.argmin(np.square(new_power - scenario['p_valid']))] 
    
    # Write and return
    #scenario['user_powers'][user_update_idx] = new_power
    
    return scenario

# Capacity reward with a negative penalty when QoS is violated
def capacity_reward(scenario):
    # Compute individual capacity
    capacity = np.log2(1 + scenario['user_sinr'])
    # Overwrite values where QoS requirement is not met
    capacity[scenario['user_sinr'] < scenario['eta']] = scenario['eta_penalty']
    
    return capacity