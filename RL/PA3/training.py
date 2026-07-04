import numpy as np
import gymnasium as gym
from GymTraffic import GymTrafficEnv


def SARSA(env, beta, Nepisodes, alpha):
    pass


def ExpectedSARSA(env, beta, Nepisodes, alpha):
    pass


def ValueFunctionSARSA(env, beta, Nepisodes, alpha):
    max_queue_length = 20
    n_actions = env.action_space.n  # 2 actions (keep or switch)

    # Initialize Value table
    V_table = np.zeros((max_queue_length + 1, max_queue_length + 1, 2, 11))

    nu = 1e-5  # Small positive number
    epsilon = 0.9
    epsilon_decay = 0.997
    min_epsilon = 0.05

    # Helper to truncate state
    def truncate_state(state):
        return [min(state[0], max_queue_length),
                min(state[1], max_queue_length),
                state[2],
                min(state[3], 10)]

    # Helper to normalize q-values
    def normalize_q_values(q_values):
        min_q = np.min(q_values)
        max_q = np.max(q_values)
        denominator = max_q - min_q + nu
        return (q_values - min_q) / denominator

    # Helper to compute q(x,a) given V(x')
    def compute_q(state, V_table):
        queue1, queue2, green, time = state
        green_idx = 0 if green == 1 else 1
        time_idx = min(time, 10)

        q_values = np.zeros(n_actions)
        for action in range(n_actions):
            env.set_state((queue1, queue2, green, time))
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_queue1 = min(next_state[0], max_queue_length)
            next_queue2 = min(next_state[1], max_queue_length)
            next_green = next_state[2]
            next_time = min(next_state[3], 10)

            next_green_idx = 0 if next_green == 1 else 1
            next_time_idx = next_time

            q_values[action] = reward + beta * V_table[next_queue1, next_queue2, next_green_idx, next_time_idx]

        return q_values

    # Helper to select action using epsilon-greedy
    def select_action(state, V_table, epsilon):
        q_values = compute_q(state, V_table)
        if np.random.random() < epsilon:
            normalized_q = normalize_q_values(q_values)
            exp_values = np.exp(normalized_q)
            action_probs = exp_values / np.sum(exp_values)
            action = np.random.choice(n_actions, p=action_probs)
        else:
            action = np.argmax(q_values)
        return action

    # Start training
    for episode in range(Nepisodes):
        state = env.reset()
        if isinstance(state, tuple):  # Handle (obs, info)
            state = state[0]
        state = truncate_state(state)

        done = False
        while not done:
            queue1, queue2, green, time = state
            green_idx = 0 if green == 1 else 1
            time_idx = min(time, 10)

            # Select action
            action = select_action(state, V_table, epsilon)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = truncate_state(next_state)

            next_queue1, next_queue2, next_green, next_time = next_state
            next_green_idx = 0 if next_green == 1 else 1
            next_time_idx = min(next_time, 10)

            # Update V_table using TD(0) update
            V_table[queue1, queue2, green_idx, time_idx] += alpha * (
                reward + beta * V_table[next_queue1, next_queue2, next_green_idx, next_time_idx]
                - V_table[queue1, queue2, green_idx, time_idx]
            )

            state = next_state

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Derive final policy
    policy = np.zeros((max_queue_length + 1, max_queue_length + 1, 2, 11), dtype=int)

    for q1 in range(max_queue_length + 1):
        for q2 in range(max_queue_length + 1):
            for green_idx in range(2):
                for time_idx in range(11):
                    green = 1 if green_idx == 0 else 2
                    state = [q1, q2, green, time_idx]
                    q_values = compute_q(state, V_table)
                    policy[q1, q2, green_idx, time_idx] = np.argmax(q_values)

    return policy




env = GymTrafficEnv() # Create and instance of the traffic controller environment.

Nepisodes = 2000   # Number of episodes to train
alpha = 0.1         # Learning rate
beta = 0.997        # Discount factor

# Learn the optimal policies using two different TD learning approaches
policy1 = SARSA(env, beta, Nepisodes, alpha)
policy2 = ExpectedSARSA(env, beta, Nepisodes, alpha)
policy3 = ValueFunctionSARSA(env, beta, Nepisodes, alpha)

# Save the policies
np.save('policy1.npy', policy1)
np.save('policy2.npy', policy2)
np.save('policy3.npy', policy3)

env.close() # Close the environment