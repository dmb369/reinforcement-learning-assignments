import numpy as np
from Assignment2Tools import prob_vector_generator, markov_matrix_generator
from itertools import product
import time

def qfunc(x, z, b, t, tau, eta, alpha, gamma, beta, P, ST, B, Delta, V, lmbda, a, Sx, Sz):
    d = np.arange(Delta + 1)
    alpha_d = alpha[d]
    b_plus = np.minimum(B, b + d)
    
    if ST == 0:
        reward = -np.square(Sx[x] - Sz[z])
    else:
        reward = -lmbda * np.square(Sx[x] - a) - (1 - lmbda) * np.square(Sx[x] - Sz[z])
    
    if t == 0:
        v1 = V[:, z, b_plus, 1]
        v2 = V[:, z, b_plus, 0]
        weighted_v = P[x, :, None] * alpha_d[None, :] * (gamma * v1 + (1 - gamma) * v2)
    else:
        t_next = t + 1 if t < tau else 0
        if ST == 1:
            wind_idx = np.where(Sx == a)[0][0] if a in Sx else x
            b_trans = np.minimum(B, b - eta + d)
            v_trans = V[:, wind_idx, b_trans, t_next]
            v_notrans = V[:, z, b_plus, t_next]
            weighted_v = P[x, :, None] * alpha_d[None, :] * (lmbda * v_trans + (1 - lmbda) * v_notrans)
        else:
            v = V[:, z, b_plus, t_next]
            weighted_v = P[x, :, None] * alpha_d[None, :] * v
    
    return reward + beta * np.sum(weighted_v)

def action_space_func(b, eta, Swind):
    actions = [[0, -1]]
    if b >= eta:
        actions.extend([1, sx] for sx in Swind)
    return actions

def immediate_reward(x, z, ST, a, lmbda):
    return np.where(ST == 0,
                   -np.square(x - z),
                   -lmbda * np.square(x - a) - (1 - lmbda) * np.square(x - z))

def policy_evaluation(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    global Sx, Sz
    Sx = np.arange(len(Swind))
    Sz = np.arange(len(Swind))

    V = np.zeros((len(Swind), len(Sz), B + 1, tau + 1))
    V_new = np.zeros((len(Swind), len(Sz), B + 1, tau + 1))

    Sb = np.arange(B + 1).astype(int)
    St = np.arange(tau + 1).astype(int)

    greedy_policy = np.zeros((len(Sx), len(Sz), B + 1, tau + 1), dtype=object)
    
    delta = np.inf
    iteration = 1

    while delta > theta or iteration <= Kmin:
        for x, z, b, t in product(range(len(Sx)), range(len(Sz)), Sb, St):
            if iteration == 1:
                actions = action_space_func(b, eta, Swind)
                max_reward = -np.inf
                best_action = None
                
                for action in actions:
                    ST, a = action
                    imm_reward = immediate_reward(Sx[x], Sz[z], ST, a, lmbda)
                    if imm_reward > max_reward:
                        max_reward = imm_reward
                        best_action = action
                
                if best_action is None and len(actions) > 0:
                    best_action = actions[0]
                    
                greedy_policy[x, z, b, t] = best_action
            
            ST, a = greedy_policy[x, z, b, t]
            q_val = qfunc(Sx[x], Sz[z], b, t, tau, eta, alpha, gamma, beta, P, ST, B, Delta, V, lmbda, a, Sx, Sz)
            V_new[x, z, b, t] = q_val

        delta = np.max(np.abs(V_new - V))
        V = np.copy(V_new)
        iteration += 1
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Delta: {delta}")

    return V


# --- System parameters ---
Swind = np.linspace(0, 1, 21)
mu_wind = 0.3
z_wind = 0.5
stddev_wind = z_wind * mu_wind
retention_prob = 0.9
P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)

lmbda = 0.7
B = 10
eta = 2
Delta = 3
mu_delta = 2
z_delta = 0.5
stddev_delta = z_delta * mu_delta
alpha = prob_vector_generator(np.arange(Delta + 1), mu_delta, stddev_delta)

tau = 4
gamma = 1 / 15
beta = 0.95
theta = 0.01
Kmin = 10

# --- Run optimized policy evaluation ---
start = time.time()
V = policy_evaluation(
    Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin
)
end = time.time()

print("Value function shape:", V.shape)
print("Sample value:", V[0, 0, 0, 0])
print('Time Taken:', (end - start), 'seconds')
