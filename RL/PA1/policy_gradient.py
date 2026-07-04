import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from CloudComputing import ServerAllocationEnv

# Feature Extraction Function
def extract_features(state):   
    num_jobs = len(state)
    
    if num_jobs == 0:
        return torch.zeros(7, dtype=torch.float32)  # Return zero vector with all features
    
    avg_priority = 0
    job_type_counts = [0, 0, 0]  # Count of jobs of each type (A, B, C)
    avg_network = 0
    avg_processing_time = 0
    
    for job in state:
        priority, job_type, network, proc_time = job
        avg_priority += priority
        
        if job_type == 'A':
            job_type_counts[0] += 1
        elif job_type == 'B':
            job_type_counts[1] += 1
        else:  
            job_type_counts[2] += 1
            
        avg_network += network
        avg_processing_time += proc_time
    
    avg_priority /= num_jobs
    avg_network /= num_jobs
    avg_processing_time /= num_jobs
    
    job_type_distribution = [count / num_jobs for count in job_type_counts]
    
    features = torch.tensor([
        num_jobs,                   
        avg_priority,             
        job_type_distribution[0],              
        job_type_distribution[1],              
        job_type_distribution[2],              
        avg_network,              
        avg_processing_time             
    ], dtype=torch.float32)
    
    return features

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Contextual Bandit Policy Gradient
def train_policy_gradient(env):
    episodes = 1440
    learning_rate = 0.001       
    input_dim = 7   
    output_dim = env.MaxServers  
    
    policy = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    epsilon_max = 0.9  
    epsilon_min = 0.01  
    decay_rate = 0.005  
    
    rewards_log = []  
    episode_rewards = []  
    
    for episode in range(episodes):
        epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * episode)
        
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            features = extract_features(state)
            action_probs = policy(features)
            
            if np.random.random() <= epsilon:
                action = random.randint(0, output_dim - 1)
            else:
                action = torch.argmax(action_probs).item()
            
            action_distribution = torch.distributions.Categorical(action_probs)
            log_prob = action_distribution.log_prob(torch.tensor(action))
            
            next_state, reward, done, truncated, _ = env.step(action + 1)
            rewards_log.append(reward)
            episode_reward += reward
            
            loss = -log_prob * reward  
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{episodes}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")

    return np.array(rewards_log), policy

# Moving Average Calculation
def moving_avg(reward_arr, window_size):
    if len(reward_arr) < window_size:
        return np.cumsum(reward_arr) / np.arange(1, len(reward_arr) + 1)
    return np.array([np.mean(reward_arr[max(0, i - window_size):i+1]) for i in range(len(reward_arr))])

# Reward Plotting
def plot_rewards(rewards, window=500):
    avg_rewards = moving_avg(rewards, window)
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards, color='blue', linewidth=2, label=f'Moving Avg (window={window})')
    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward")
    plt.title("Contextual Bandit Policy Gradient: Moving Avg Rewards")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Environment Setup
env = ServerAllocationEnv()
rewards, trained_policy = train_policy_gradient(env)
plot_rewards(rewards)

print("Training complete.")
print(np.average(rewards[-100:]))

env.close()
