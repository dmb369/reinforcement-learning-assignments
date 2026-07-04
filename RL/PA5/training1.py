import numpy as np
import gymnasium as gym
import pygame
# Write just ONE line of code below this comment to import DQN from stable baseline
from stable_baselines3 import DQN

def visualize_model_performance(model):
    env = gym.make('MountainCar-v0', render_mode='human')
    
    x, _ = env.reset()
    total_reward = 0
    terminated, truncated = False, False
    while not(terminated) and not(truncated):
        action, _ = model.predict(x)
        x, reward, terminated, truncated, _ = env.step(action)        
        total_reward += reward
    
    print('Total reward = {}'.format(total_reward))
    env.close()
    # pygame.display.quit() # Use this line when the display screen is not going away
        

# Initiate the mountain car environment.
env = gym.make('MountainCar-v0')


# Write just TWO lines of code below this comment to train a DQN model for mountain car.
model = DQN("MlpPolicy", env, verbose=1, train_freq=4, buffer_size=100000, learning_rate=1e-3, batch_size=128, target_update_interval=1000, tau=1.0, learning_starts=2000, exploration_initial_eps=1.0, exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[128, 128]))
model.learn(total_timesteps=200000)
# Close the mountain car environment.
env.close


# Write just ONE line of code below to save the DQN model that you have trained. YOU DON'T HAVE TO SUBMIT THIS MODEL.
model.save("dqn_mountaincar")

# Write just ONE line of code below this comment to call visualize_model_performance in order to test the performance of the trained model
#for i in range(10):
    #visualize_model_performance(model)