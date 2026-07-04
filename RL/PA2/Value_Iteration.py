import numpy as np
from Assignment2Tools import prob_vector_generator, markov_matrix_generator
from itertools import product
import time

def qfunc(x, z, b, t, tau, eta, alpha, gamma, beta, P, ST, B, Delta, V, lmbda, a, Sx, Sz):
    """Vectorized Q-function for policy evaluation with broadcasting"""
    d = np.arange(Delta + 1)
    alpha_d = alpha[d]
    b_plus = np.minimum(B, b + d)  # Broadcast over delta
    
    # Base reward calculation
    if ST == 0:
        reward = -np.square(Sx[x] - Sz[z])
    else:
        reward = -lmbda * np.square(Sx[x] - a) - (1 - lmbda) * np.square(Sx[x] - Sz[z])
    
    # Value function component
    if t == 0:
        v1 = V[:, z, b_plus, 1]  # (len(Sx), Delta+1)
        v2 = V[:, z, b_plus, 0]  # (len(Sx), Delta+1)
        weighted_v = P[x, :, None] * alpha_d[None, :] * (gamma * v1 + (1 - gamma) * v2)
    else:
        t_next = t + 1 if t < tau else 0
        if ST == 1:
            wind_idx = np.where(Sx == a)[0][0] if a in Sx else x
            b_trans = np.minimum(B, b - eta + d)
            v_trans = V[:, wind_idx, b_trans, t_next]  # (len(Sx), Delta+1)
            v_notrans = V[:, z, b_plus, t_next]        # (len(Sx), Delta+1)
            weighted_v = P[x, :, None] * alpha_d[None, :] * (lmbda * v_trans + (1 - lmbda) * v_notrans)
        else:
            v = V[:, z, b_plus, t_next]  # (len(Sx), Delta+1)
            weighted_v = P[x, :, None] * alpha_d[None, :] * v
    
    return reward + beta * np.sum(weighted_v)

def action_space_func(b, eta, Swind):
    """Generate action space with vectorized options"""
    actions = [[0, -1]]  # Default no transmission action
    if b >= eta:
        # Vectorized transmission actions
        actions.extend([1, sx] for sx in Swind)
    return actions

def value_iteration(Swind, Sz_grid, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin):
   global Sx, Sz
   Sx = Swind
   Sz = Sz_grid

   Sb = np.arange(B + 1)
   St = np.arange(tau + 1)
   

   # Initialize V arbitrarily (to zeros)
   V = np.zeros((len(Sx), len(Sz), B + 1, tau + 1))
   policy = np.zeros((len(Sx), len(Sz), B + 1, tau + 1), dtype=object)

   # Initialize delta = inf
   delta = np.inf
   iteration = 1

   # Main value iteration loop
   while delta > theta or iteration <= Kmin:
       print(f"Value Iteration: {iteration}, Delta: {delta:.4f}")
       
       # Create a new V for this iteration
       V_new = np.zeros_like(V)
       
       # For all states
       for x_idx, z_idx, b, t in product(range(len(Sx)), range(len(Sz)), Sb, St):
           # Get available actions for this state
           actions = action_space_func(b, eta, Sx)
           
           # Find max Q-value for state
           max_q = -np.inf
           best_action = None
           
           for action in actions:
               ST,a = action
               q_val = qfunc(x_idx, z_idx, b, t, tau, eta, alpha, gamma, beta,
                                       P, ST, B, Delta, V, lmbda, a, Sx, Sz)
               if q_val > max_q:
                   max_q = q_val
                   best_action = action
           
           # Update V_new and policy
           V_new[x_idx, z_idx, b, t] = max_q
           policy[x_idx, z_idx, b, t] = best_action
       
       # Compute delta as max absolute difference
       delta = np.max(np.abs(V_new - V))
       
       # Update V
       V = V_new.copy()
       
       iteration += 1

   return policy, V

# === Main Run ===
# Parameters
Swind = np.linspace(0, 1, 21)
Sz = np.linspace(0, 1, 21)

mu_wind = 0.3
z_wind = 0.5
stddev_wind = z_wind * mu_wind
retention_prob = 0.9

Delta = 3
mu_delta = 2
z_delta = 0.5
stddev_delta = z_delta * mu_delta

P = markov_matrix_generator(Swind, mu_wind, stddev_wind, retention_prob)
alpha = prob_vector_generator(np.arange(Delta+1), mu_delta, stddev_delta)

lmbda = 0.7
B = 10
eta = 2
tau = 4
gamma = 1/15
beta = 0.95
theta = 0.01
Kmin = 10

print("\nRunning Optimized Value Iteration...")
start = time.time()
policy, V = value_iteration(
    Swind, Sz, P, lmbda, B, eta, Delta, alpha, tau, gamma, beta, theta, Kmin
)
end = time.time()
print("Sample value:", V[0, 0, 0, 0])
print('Time Taken:', (end - start), 'seconds')
print("\nFinished.")
