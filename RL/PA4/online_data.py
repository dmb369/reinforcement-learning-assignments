import numpy as np
import pandas as pd
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

def load_offline_data(path, min_score=-float('inf')):    
    state_data = []
    action_data = []
    reward_data = []
    next_state_data = []
    terminated_data = []
    dataset = pd.read_csv(path)
    dataset_group = dataset.groupby('Play #')
    for play_no, df in dataset_group:
        start = 0
        if isinstance(df.iloc[0, 1], str) and '{}' in df.iloc[0, 1]:
            start = 1
        df = df[start:]
        state = []
        for s in df.iloc[:, 1]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                state.append([float(val.strip(',')) for val in s])
            else:
                state.append(s)
        state = np.array(state)
        action = np.array(df.iloc[:, 2]).astype(int)
        reward = np.array(df.iloc[:, 3]).astype(np.float32)
        next_state = []
        for s in df.iloc[:, 4]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                next_state.append([float(val.strip(',')) for val in s])
            else:
                next_state.append(s)
        next_state = np.array(next_state)
        terminated = np.array(df.iloc[:, 5]).astype(int)
        total_reward = np.sum(reward)
        
        if total_reward >= min_score:
            state_data.append(state)
            action_data.append(action)
            reward_data.append(reward)
            next_state_data.append(next_state)
            terminated_data.append(terminated)
            
    if not state_data:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    state_data = np.concatenate(state_data)
    action_data = np.concatenate(action_data)
    reward_data = np.concatenate(reward_data)
    next_state_data = np.concatenate(next_state_data)
    terminated_data = np.concatenate(terminated_data)
    
    return state_data, action_data, reward_data, next_state_data, terminated_data

def plot_reward(total_reward_per_episode, window_length, title="Training Progress"):
    plt.figure(figsize=(10, 5))
    plt.plot(total_reward_per_episode, label='Total Reward per Episode', alpha=0.6)
    if len(total_reward_per_episode) > window_length:
        weights = np.ones(window_length) / window_length
        moving_avg = np.convolve(total_reward_per_episode, weights, mode='valid')
        padding = np.full(window_length-1, np.nan)
        moving_avg = np.concatenate([padding, moving_avg])
    else:
        moving_avg = np.cumsum(total_reward_per_episode) / np.arange(1, len(total_reward_per_episode) + 1)
    plt.plot(moving_avg, label=f'Moving Average (window={window_length})', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def build_DQN(state_dim, action_dim):
    model = Sequential([
        Input(shape=(state_dim + action_dim,)),  
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1, activation="linear")  
    ])
    return model

def normalize_q_values(q_values):
    min_q = np.min(q_values)
    max_q = np.max(q_values)
    nu = 1e-6
    return (q_values - min_q) / (max_q - min_q + nu)

def exploration_prob_scheduler(episode, Nepisodes=1000, min_eps=0.05, max_eps=1.0, decay_rate=5.0):
    return min_eps + (max_eps - min_eps) * np.exp(-decay_rate * episode / Nepisodes)

def select_action(state, model, n_actions, epsilon, state_mean, state_std):
    state_norm = (state - state_mean) / state_std
    actions_one_hot = np.eye(n_actions)
    state_batch = np.tile(state_norm, (n_actions, 1))
    input_tensor = np.concatenate([state_batch, actions_one_hot], axis=1)
    q_values = model.predict(input_tensor, verbose=0).flatten()
    
    if np.random.random() < epsilon:
        normalized_q = normalize_q_values(q_values)
        exp_values = np.exp(normalized_q)
        action_probs = exp_values / np.sum(exp_values)
        action = np.random.choice(n_actions, p=action_probs)
    else:
        action = np.argmax(q_values)
    
    return action

def add_to_buffer(x, a, r, x_dash, terminated, state_buffer, action_buffer, reward_buffer, 
                 next_state_buffer, terminated_buffer, buffer_size, buffer_counter, buffer_ix):
    state_buffer[buffer_ix] = x
    action_buffer[buffer_ix] = a
    reward_buffer[buffer_ix] = r
    next_state_buffer[buffer_ix] = np.zeros_like(x) if terminated else x_dash
    terminated_buffer[buffer_ix] = int(terminated)
    
    buffer_counter = min(buffer_counter + 1, buffer_size)
    buffer_ix = (buffer_ix + 1) % buffer_size
        
    return buffer_counter, buffer_ix

def generate_training_data(state_buffer, action_buffer, reward_buffer, next_state_buffer, terminated_buffer, 
                       buffer_counter, model_predict, model_target, Nb, beta, n_actions, exploration_prob, state_mean, state_std):
    ix = np.random.choice(buffer_counter, size=Nb, replace=False)
    state_batch = state_buffer[ix]
    action_batch = action_buffer[ix]
    reward_batch = reward_buffer[ix]
    next_state_batch = next_state_buffer[ix]
    terminated_batch = terminated_buffer[ix]
    
    state_batch_norm = (state_batch - state_mean) / state_std
    next_state_batch_norm = (next_state_batch - state_mean) / state_std
    
    action_one_hot_batch = np.zeros((Nb, n_actions))
    action_one_hot_batch[np.arange(Nb), action_batch] = 1
    
    X = np.concatenate([state_batch_norm, action_one_hot_batch], axis=1)
    y = np.copy(reward_batch)
    
    non_terminal_indices = np.where(terminated_batch == 0)[0]
    if len(non_terminal_indices) > 0:
        next_states = next_state_batch_norm[non_terminal_indices]
        batch_size = len(non_terminal_indices)
        next_states_repeated = np.repeat(next_states, n_actions, axis=0)
        actions_one_hot = np.tile(np.eye(n_actions), (batch_size, 1))
        input_tensor = np.concatenate([next_states_repeated, actions_one_hot], axis=1)
        
        next_q_values = model_target.predict(input_tensor, verbose=0).reshape(batch_size, n_actions)
        greedy_actions = np.argmax(next_q_values, axis=1)
        
        normalized_q = normalize_q_values(next_q_values)
        exp_values = np.exp(normalized_q)
        softmax_probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        action_probs = exploration_prob * softmax_probs
        action_probs[np.arange(batch_size), greedy_actions] += (1 - exploration_prob)
        
        expected_q_values = np.sum(next_q_values * action_probs, axis=1)
        y[non_terminal_indices] = reward_batch[non_terminal_indices] + beta * expected_q_values
    
    return X, y

def initialize_buffer_with_offline_data(offline_data, buffer_size, state_dim, estimate_normalization=True):
    state_data, action_data, reward_data, next_state_data, terminated_data = offline_data
    n_samples = len(state_data)
    
    if n_samples == 0:
        print("No offline data available that meets the minimum score criterion")
        return (np.zeros((buffer_size, state_dim)), np.zeros(buffer_size, dtype=np.uint8), 
                np.zeros(buffer_size), np.zeros((buffer_size, state_dim)), 
                np.zeros(buffer_size, dtype=np.uint8), 0, 0, 
                np.zeros(state_dim), np.ones(state_dim))
    
    state_buffer = np.zeros((buffer_size, state_dim))
    action_buffer = np.zeros(buffer_size, dtype=np.uint8)
    reward_buffer = np.zeros(buffer_size)
    next_state_buffer = np.zeros((buffer_size, state_dim))
    terminated_buffer = np.zeros(buffer_size, dtype=np.uint8)
    
    n_samples_to_use = min(n_samples, buffer_size)
    state_buffer[:n_samples_to_use] = state_data[:n_samples_to_use]
    action_buffer[:n_samples_to_use] = action_data[:n_samples_to_use]
    reward_buffer[:n_samples_to_use] = reward_data[:n_samples_to_use]
    next_state_buffer[:n_samples_to_use] = next_state_data[:n_samples_to_use]
    terminated_buffer[:n_samples_to_use] = terminated_data[:n_samples_to_use]
    
    buffer_counter = n_samples_to_use
    buffer_ix = n_samples_to_use % buffer_size
    
    if estimate_normalization and buffer_counter > 0:
        state_mean = np.mean(state_buffer[:buffer_counter], axis=0)
        state_std = np.std(state_buffer[:buffer_counter], axis=0) + 1e-8
    else:
        state_mean = np.zeros(state_dim)
        state_std = np.ones(state_dim)
    
    print(f"Initialized replay buffer with {buffer_counter} samples from offline data")
    return state_buffer, action_buffer, reward_buffer, next_state_buffer, terminated_buffer, buffer_counter, buffer_ix, state_mean, state_std

def DQN_training(env, offline_data=None, use_offline_data=False, min_score=-float('inf'), E=20):
    Nu = 1                 # Predict DQN training interval
    Nb = 64                # Batch size for training
    Nt = 100               # Target network update interval
    beta = 0.99            # Discount factor
    Nepisodes = 1000       # Number of episodes to train
    alpha = 0.001          # Learning rate
    Nsave = 50             # Save model interval
    buffer_size = 50000    # Replay buffer size
    
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    model_predict = build_DQN(state_dim, n_actions)
    model_target = build_DQN(state_dim, n_actions)
    
    optimizer = Adam(learning_rate=alpha)
    model_predict.compile(loss='mse', optimizer=optimizer)
    model_target.set_weights(model_predict.get_weights())
    
    
    if use_offline_data and offline_data is not None:
        state_buffer, action_buffer, reward_buffer, next_state_buffer, terminated_buffer, buffer_counter, buffer_ix, state_mean, state_std = \
            initialize_buffer_with_offline_data(offline_data, buffer_size, state_dim, estimate_normalization=True)
    else:
        state_buffer = np.zeros((buffer_size, state_dim))
        action_buffer = np.zeros(buffer_size, dtype=np.uint8)
        reward_buffer = np.zeros(buffer_size)
        next_state_buffer = np.zeros((buffer_size, state_dim))
        terminated_buffer = np.zeros(buffer_size, dtype=np.uint8)
        buffer_counter = 0
        buffer_ix = 0
        state_mean = np.zeros(state_dim)
        state_std = np.ones(state_dim)
    
    counter_save = 0
    counter_target = 0
    counter_predict = 0
    
    total_reward_per_episode = []
    last_update_time = time.time()

    for episode in range(Nepisodes):
        epsilon = exploration_prob_scheduler(episode)
        
        x, _ = env.reset()
        
        total_reward = 0
        end_episode = False
        k = 0
        
        while not end_episode:
            a = select_action(x, model_predict, n_actions, epsilon, state_mean, state_std)
            x_dash, r, terminated, truncated, _ = env.step(a)
            total_reward += r
            
            if not use_offline_data or episode >= E:
                buffer_counter, buffer_ix = add_to_buffer(
                    x, a, r, x_dash, terminated, 
                    state_buffer, action_buffer, reward_buffer, 
                    next_state_buffer, terminated_buffer, 
                    buffer_size, buffer_counter, buffer_ix
                )
                
            counter_predict += 1
            if counter_predict >= Nu:
                if buffer_counter >= Nb:  
                    if buffer_counter > 0:
                        state_mean = np.mean(state_buffer[:buffer_counter], axis=0)
                        state_std = np.std(state_buffer[:buffer_counter], axis=0) + 1e-8
                    
                    X, y = generate_training_data(
                        state_buffer, action_buffer, reward_buffer, 
                        next_state_buffer, terminated_buffer, 
                        buffer_counter, model_predict, model_target, 
                        Nb, beta, n_actions, epsilon, state_mean, state_std
                    )
                    
                    model_predict.train_on_batch(X, y)

                counter_predict = 0
                counter_target += 1            
                if counter_target >= Nt:
                    model_target.set_weights(model_predict.get_weights())
                    counter_target = 0
                
                counter_save += 1
                if counter_save >= Nsave:
                    model_name = 'DQN_offline_true.keras' if use_offline_data else 'DQN_offline_false.keras'
                    model_predict.save(model_name)
                    counter_save = 0
            
            current_time = time.time()
            if current_time - last_update_time >= 60:
                print(f"[MINUTE UPDATE] Episode: {episode+1}, Step: {k}, Current reward: {total_reward:.2f}, Buffer size: {buffer_counter}/{buffer_size}")
                last_update_time = current_time
            
            x = np.copy(x_dash)
            k += 1
            
            if k > 800 or terminated or truncated:
                end_episode = True
                
        total_reward_per_episode.append(total_reward)
        
        print(f'Episode = {episode+1}, Total reward = {np.round(total_reward, 2)}, Epsilon = {epsilon:.3f}, Offline Data = {use_offline_data}')
        if use_offline_data:
            print(f'Data collection: {"Not collecting online data yet" if episode < E else "Collecting online data"}')
    
    model_name = 'DQN_offline_true.keras' if use_offline_data else 'DQN_offline_false.keras'
    model_predict.save(model_name)
    
    return model_predict, np.array(total_reward_per_episode)

env = gym.make('MountainCar-v0', render_mode="human")
path = 'car_dataset.csv'
min_score = -np.inf
    
offline_data = load_offline_data(path, min_score)
window_length = 50 
    
print("Training DQN without offline data...")
final_model_no_offline, rewards_no_offline = DQN_training(env, offline_data=None, use_offline_data=False)
plot_reward(rewards_no_offline, window_length, "DQN Training (No Offline Data)")
    
'''print("Training DQN with offline data...")
final_model_with_offline, rewards_with_offline = DQN_training(env, offline_data, use_offline_data=True)
plot_reward(rewards_with_offline, window_length, "DQN Training (With Offline Data)")'''
    
env.close()