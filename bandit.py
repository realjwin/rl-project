import copy
import numpy as np
import tqdm as tqdm

from aux_functions import *

def run_bandit(num_iterations, num_steps, initialized_scenario, power_adjust, 
               reward_function, bandit_strategy, alpha, epsilon, step_size, 
               move_strategy, move_steps, max_user_box, reset, move_stepsize=1):

    scenario = copy.deepcopy(initialized_scenario)
    
    num_users = scenario['num_users']
    
    # Setup tracking variables
    avg_learning_reward_hist = np.zeros((num_steps,num_users))
    avg_actual_reward_hist = np.zeros((num_steps,num_users))
    sinr_hist = []
    user_power_hist = []
    
    final_actual_reward = []
    final_user_power = []
    user_dist = []
    
    for iter_num in tqdm.tqdm(range(num_iterations)):
        #np.random.seed(global_seed+iter_num)
        scenario = copy.deepcopy(initialized_scenario)
        
        # Set user and bs location
        scenario = set_locations(scenario, max_user_box)
        
        ## Time evolution
        learning_reward_list = []
        actual_reward_list = []
        sinr_list = []
        user_power_list = []
        user_idx_list = []
        
        # For each timestep
        for time_idx in range(num_steps):
            
            # Increment user locations
            if move_strategy != 'stationary':
                if np.mod(time_idx+1, move_steps) == 0:
                    if move_strategy == 'random':
                        scenario = move_random(scenario, move_stepsize)
                    elif move_strategy == 'box':
                        scenario = move_random_boxed(scenario)
                    if reset:
                        scenario = reset_users(scenario)
            
            # Update SINR values
            scenario = update_sinr(scenario)
            
            sinr_list.append(scenario['user_sinr'])

            user_idx = pick_user(scenario, bandit_strategy, epsilon, time_idx, c=1)

            user_idx_list.append(user_idx)
            
            # Send SINR update to a specific user
            scenario = user_power_adjust(scenario, user_idx, power_adjust, step_size=step_size)
            
            user_power = scenario['p_valid'][scenario['user_power_idx']]
            
            user_power_list.append(user_power)
            
            # Receive new sum-rate as reward
            learning_reward, actual_reward = reward(scenario, reward_function)
            
            # Store metrics
            learning_reward_list.append(learning_reward)
            actual_reward_list.append(actual_reward)
            
            # Update action-value estimate
            scenario = update_estimate(scenario, learning_reward, bandit_strategy, time_idx, alpha)
        
        # Convert to arrays
        avg_learning_reward_hist += np.asarray(learning_reward_list)
        avg_actual_reward_hist += np.asarray(actual_reward_list)
        sinr_hist.append(np.asarray(sinr_list))
        user_power_hist.append(np.asarray(user_power_list))
        user_idx_list += np.asarray(user_idx_list)
    
        final_actual_reward.append(actual_reward)
        final_user_power.append(user_power)
        user_dist.append(np.sqrt(np.sum(np.square(scenario['user_locations'] - scenario[
                'bs_location']), axis=-1)))
    
    # Track where failures are happening, mostly for testing
    final_actual_reward = np.asarray(final_actual_reward)
    final_user_power = np.asarray(final_user_power)
    user_dist = np.asarray(user_dist)
    
    #failed_reward = final_actual_reward[np.sum(final_actual_reward, axis = -1) <= 1]
    #failed_power = final_user_power[np.sum(final_actual_reward, axis = -1) <= 1]
    #failed_dist = user_dist[np.sum(final_actual_reward, axis = -1) <= 1]
    
    # Compute total rewards
    avg_learning_reward_hist = np.sum(avg_learning_reward_hist, axis=1) / num_iterations
    avg_actual_reward_hist = np.sum(avg_actual_reward_hist, axis=1) / num_iterations
    
    return avg_actual_reward_hist, avg_learning_reward_hist, np.asarray(user_power_hist), np.asarray(sinr_hist)