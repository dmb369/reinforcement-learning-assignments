mport numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from CloudComputing import ServerAllocationEnv

def extract_features(obsv):   
    """ Extracts feature vectors from the environment observation."""
    n_jobs = len(obsv)
    
    if n_jobs == 0:
        return np.zeros(7)  # Return zero vector with all features
    
    avg_priority = 0
    type_counts = [0, 0, 0]  # Count of jobs of each type (A, B, C)
    avg_network = 0
    avg_proc_time = 0
    
    for job in obsv:
        priority, job_type, network, proc_time = job
        avg_priority += priority
        
        if job_type == 'A':
            type_counts[0] += 1
        elif job_type == 'B':
            type_counts[1] += 1
        else:  
            type_counts[2] += 1
            
        avg_network += network
        avg_proc_time += proc_time
    
    avg_priority /= n_jobs
    avg_network /= n_jobs
    avg_proc_time /= n_jobs
    
    type_job = [count / n_jobs for count in type_counts]
    
    features = np.array([
        n_jobs,                   
        avg_priority,             
        type_job[0],              
        type_job[1],              
        type_job[2],              
        avg_network,              
        avg_proc_time             
    ])
    
    return features

def UCBPolicy(env, delta=0.01, epsilon_max=1.0, epsilon_min=0.01, decay_rate=0.005):
    """ Upper Confidence Bound (UCB) algorithm for linear bandits with decaying epsilon."""
    Nfeatures = 7  
    Nactions = env.MaxServers  
    theta_hat = np.zeros((Nactions, Nfeatures + 1))  
    A = np.array([np.eye(Nfeatures + 1) for _ in range(Nactions)])
    b = np.zeros((Nactions, Nfeatures + 1))
    
    Nepisodes = 1440
    reward_history = []
    
    for episode in range(Nepisodes):
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * episode)  # Exponential decay
        obsv, _ = env.reset()
        truncated = False
        
        while not truncated:
            features = extract_features(obsv)
            z_t = np.append(features, 1.0)  
            
            ucb_values = []
            for action in range(Nactions):
                A_inv = np.linalg.inv(A[action])
                theta_action = np.dot(A_inv, b[action])
                
                confidence_bound = epsilon * np.sqrt(np.dot(z_t, np.dot(A_inv, z_t)))
                ucb_values.append(np.dot(theta_action, z_t) + confidence_bound)
            
            action = np.argmax(ucb_values)
            
            next_obsv, reward, _, truncated, _ = env.step(action + 1)
            reward_history.append(reward)
            
            if not truncated:
                z_t_temp = z_t.reshape(-1, 1)
                A[action] += np.dot(z_t_temp, z_t_temp.T)
                b[action] += reward * z_t_temp.flatten()
                theta_hat[action] = np.dot(np.linalg.inv(A[action] + delta * np.eye(Nfeatures + 1)), b[action])
                
                obsv = next_obsv
    
    return np.array(reward_history), theta_hat

def plot_rewards(rewards, window=500):
    avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10,5))
    plt.plot(avg_rewards, color='blue', label=f'Receding Window Avg (window={window})')
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward")
    plt.title("UCB Linear Bandit: Reward Progression")
    plt.legend()
    plt.show()

# Create environment 
env = ServerAllocationEnv()

rewards, theta_hat = UCBPolicy(env)
plot_rewards(rewards)
env.close()

print("Training complete.")