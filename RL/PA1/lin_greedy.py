import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from CloudComputing import ServerAllocationEnv

def extract_features(obsv):   
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

def LinGreedyPolicy(env):
    Nfeatures = 7  
    Nactions = env.MaxServers  
    theta = np.zeros((Nactions, Nfeatures+1))  
    A = np.zeros((Nactions, Nfeatures+1, Nfeatures+1))
    b = np.zeros((Nactions, Nfeatures+1))
    
    epsilon_max = 0.9
    epsilon_min = 0.01  
    decay_rate = 0.005
    
    Nepisodes = 2000
    reward_history = []
    
    for episode in range(Nepisodes):
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * episode)  
        obsv, _ = env.reset()
        truncated = False
        
        while not truncated:
            features = extract_features(obsv)
            z = np.append(features, 1.0)  
            if np.random.random() <= epsilon:
                action = random.randint(0, Nactions-1)
            else:
                action = np.argmax(np.dot(theta, z))
            
            next_obsv, reward, _, truncated, _ = env.step(action + 1)
            reward_history.append(reward)
            
            if not truncated:
                z_temp = z.reshape(-1, 1)
                A[action] += np.matmul(z_temp, np.transpose(z_temp))
                b[action] += reward * z_temp.reshape(-1)
                theta[action] = np.dot(np.linalg.inv(A[action] + 0.01 * np.eye(Nfeatures + 1)), b[action])
                obsv = next_obsv
    
    return np.array(reward_history), theta

def receding_window_avg(reward_arr, window_size):
    if len(reward_arr) < window_size:
        return np.cumsum(reward_arr) / np.arange(1, len(reward_arr) + 1)
    return np.array([np.mean(reward_arr[max(0, i - window_size):i+1]) for i in range(len(reward_arr))])

def plot_rewards(rewards, window=500):
    avg_rewards = receding_window_avg(rewards, window)
    plt.figure(figsize=(10,5))
    plt.plot(avg_rewards, color='red', label=f'Receding Window Avg (window={window})')
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward")
    plt.title("Receding Window Time Average of Rewards")
    plt.legend()
    plt.show()


env = ServerAllocationEnv()
rewards, theta = LinGreedyPolicy(env)
plot_rewards(rewards)

env.close()
print("Training complete.")
print(np.average(rewards[-100:]))