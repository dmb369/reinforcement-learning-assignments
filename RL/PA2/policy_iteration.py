import numpy as np
from itertools import product
import time
from Assignment2Tools import prob_vector_generator, markov_matrix_generator

def qfunc(x, z, b, t, tau, eta, alpha, gamma, beta, P, ST, B_max, Delta, V, lmbda, a_idx, Sx, Sz):
    d = np.arange(Delta + 1)
    alpha_d = alpha[d]
    b_plus = np.minimum(B_max, b + d)

    if ST == 0:
        reward = -np.square(Sx[x] - Sz[z])
    else:
        a = Sx[a_idx]
        reward = -lmbda * np.square(Sx[x] - a) - (1 - lmbda) * np.square(Sx[x] - Sz[z])

    if t == 0:
        v1 = V[:, z, b_plus, 1]
        v2 = V[:, z, b_plus, 0]
        weighted_v = P[x, :, None] * alpha_d[None, :] * (gamma * v1 + (1 - gamma) * v2)
    else:
        t_next = t + 1 if t < tau else 0
        if ST == 1:
            b_trans = np.minimum(B_max, b - eta + d)
            v_trans = V[:, a_idx, b_trans, t_next]
            v_notrans = V[:, z, b_plus, t_next]
            weighted_v = P[x, :, None] * alpha_d[None, :] * (lmbda * v_trans + (1 - lmbda) * v_notrans)
        else:
            v = V[:, z, b_plus, t_next]
            weighted_v = P[x, :, None] * alpha_d[None, :] * v

    return reward + beta * np.sum(weighted_v)

def action_space_func(b, eta, Swind):
    action_space = [[0, -1]]
    if b >= eta:
        for sx in range(len(Swind)):
            action_space.append([1, sx])
    return action_space

def policy_evaluation(policy, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    Sx = np.arange(len(Swind))
    Sz = np.arange(len(Swind))
    V = np.zeros((len(Sx), len(Sz), B + 1, tau + 1))
    V_new = np.zeros_like(V)
    Sb = np.arange(B + 1).astype(int)
    St = np.arange(tau + 1).astype(int)
    delta_val = np.inf
    iteration = 1

    while delta_val > theta or iteration <= Kmin:
        for x_idx, z_idx, b, t in product(range(len(Sx)), range(len(Sz)), Sb, St):
            ST, a_idx = policy[x_idx, z_idx, b, t]
            q_val = qfunc(x_idx, z_idx, b, t, tau, eta, alpha, gamma, beta, P, ST, B, Delta, V, lmbda, a_idx, Sx, Sz)
            V_new[x_idx, z_idx, b, t] = q_val

        delta_val = np.max(np.abs(V_new - V))
        V = np.copy(V_new)

        if iteration % 10 == 0:
            print(f"    PE Iteration {iteration}: Delta = {delta_val:.6f}")

        iteration += 1

    return V

def policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
    Sx = np.arange(len(Swind))
    Sz = np.arange(len(Swind))
    policy = np.zeros((len(Sx), len(Sz), B + 1, tau + 1), dtype=object)

    for x_idx, z_idx, b, t in product(range(len(Sx)), range(len(Sz)), range(B + 1), range(tau + 1)):
        actions = action_space_func(b, eta, Swind)
        policy[x_idx, z_idx, b, t] = actions[0]

    V = np.zeros((len(Sx), len(Sz), B + 1, tau + 1))
    converged = False
    iteration = 1

    while not converged:
        print(f"Policy Iteration {iteration}:")
        print(f"  - Running policy evaluation...")
        V = policy_evaluation(policy, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)

        print(f"  - Running policy improvement...")
        policy_stable = True
        policy_changes = 0

        for x_idx, z_idx, b, t in product(range(len(Sx)), range(len(Sz)), range(B + 1), range(tau + 1)):
            actions = action_space_func(b, eta, Swind)
            current_ST, current_a_idx = policy[x_idx, z_idx, b, t]
            current_value = V[x_idx, z_idx, b, t]
            max_value = current_value
            best_action = (current_ST, current_a_idx)

            for ST, a_idx in actions:
                q_val = qfunc(x_idx, z_idx, b, t, tau, eta, alpha, gamma, beta, P, ST, B, Delta, V, lmbda, a_idx, Sx, Sz)
                if q_val > max_value + 1e-9:
                    max_value = q_val
                    best_action = (ST, a_idx)

            if best_action != (current_ST, current_a_idx):
                policy_stable = False
                policy_changes += 1
                policy[x_idx, z_idx, b, t] = best_action

        print(f"  - Policy changes: {policy_changes} states")

        if policy_stable:
            print(f"  - Policy is stable. Converged!")
            converged = True
        else:
            print(f"  - Policy is not stable. Moving to next iteration.")

        print()
        iteration += 1

    print("Running final policy evaluation...")
    V = policy_evaluation(policy, Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)

    return policy, V

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

start = time.time()
optimal_policy, optimal_V = policy_iteration(Swind, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin)
end = time.time()

print("Execution time:", end - start, "seconds")
print("Value function shape:", optimal_V.shape)
print("Policy shape:", optimal_policy.shape)
print("Sample of value function at state (0,0,0,0):", optimal_V[0, 0, 0, 0])
print("Sample of policy at state (0,0,5,0):", optimal_policy[0, 0, 5, 0])
print(optimal_policy)
