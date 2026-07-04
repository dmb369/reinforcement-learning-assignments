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
        

class Custom_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        position = observation[0]
        velocity = observation[1]

        if terminated:
            modified_reward = 1000  # Large reward for actually reaching the goal
        else:
            # Component 1: Position reward - exponential to heavily favor rightward movement
            position_component = 10 * (position + 1.2) ** 2  # Quadratic growth
            
            # Component 2: Speed reward - but only if moving in beneficial direction
            speed_component = abs(velocity) * 5
            
            # Component 3: Strong time penalty to force completion
            time_penalty = -5.0  # Much stronger penalty
            
            # Bonus for being close to goal
            if position > 0.3:  # Very close to goal at 0.5
                position_component += 50
            
            modified_reward = position_component + speed_component + time_penalty


        
        return observation, modified_reward, terminated, truncated, info


# Initiate the mountain car environment.
env = gym.make('MountainCar-v0')


# Write just one line of code below this comment to create a modified environment using Custom_Wrapper class.
env = Custom_Wrapper(env) #try human with "" render

# Write just TWO lines of code below this comment to train a DQN model for mountain car.
model = DQN("MlpPolicy", env, verbose=1, train_freq=4, buffer_size=100000, learning_rate=1e-3, batch_size=128, target_update_interval=1000, tau=1, learning_starts=2000, exploration_initial_eps=1.0, exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[128,128]))
model.learn(total_timesteps=200000)
# Close the mountain car environment.
env.close


# Write just ONE line of code below to save the DQN model that you have trained.
model.save("MODEL2")
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL2.
# THE MODEL THAT YOU SUBMIT MUST NOT EXCEED 1 MB. ELSE ZERO FOR SECTION 4.

# Write just ONE line of code below this comment to call visualize_model_performance in order to test the performance of the trained model
visualize_model_performance(model)