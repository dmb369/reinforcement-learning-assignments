import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


class ModifiedRobotNavigationEnv(gym.Wrapper):
    def __init__(self, env, H):
        super().__init__(env)
        self.H = H   # Number of time slots in a mini episode.
        
        # The action space of the modified robot navigation environment.
        Delta = H*self.env.delta
        self.action_space = gym.spaces.Box(-Delta*np.ones(2), Delta*np.ones(2), dtype=np.float32)
        
    
    def conventional_policy(self, robot_position, goal_intermediate):
        tolerance = 1e-6  # small enough for most grid-based movements
        
        x_current, y_current = robot_position
        x_goal, y_goal = goal_intermediate
        
        if (x_goal - x_current) > tolerance:
            return 3  # right
        elif (x_current - x_goal) > tolerance:
            return 2  # left
        elif (y_goal - y_current) > tolerance:
            return 0  # up
        elif (y_current - y_goal) > tolerance:
            return 1  # down
        else:
            return -1  # don't move



    def step(self, action):
        # Compute the intermediate goal
        goal_intermediate = self.env.robot_position + action        
        grid_x = int(goal_intermediate[0]/self.env.delta) + 1
        grid_y = int(goal_intermediate[1]/self.env.delta) + 1        
        grid_x = self.env.delta*(0.5 + (grid_x - 1))
        grid_y = self.env.delta*(0.5 + (grid_y - 1))
        goal_intermediate = np.array([grid_x, grid_y])
        
        # Simulate a mini-episode using the conventional policy.
        reward_miniepisode = 0
        for h in range(self.H):
            reward = -np.sqrt(np.sum((self.env.robot_position - self.env.goal)**2))
            
            a = self.conventional_policy(self.env.robot_position, goal_intermediate)  # Call conventional policy.
            
            if a!=-1: # If a==-1, then robot position does not change.
                self.env.robot_position = self.env.robot_position + self.env.action_dict[a+0]*self.env.delta
            
            self.env.trail.append(self.env.robot_position)
            
            # Check for collision
            terminated = self.env.check_collision()
            if terminated:
                reward = -10000
            
            # Check if goal is reached
            if not(terminated):
                if (self.env.goal[0] - self.env.robot_position[0])**2 + (self.env.goal[1] - self.env.robot_position[1])**2<=self.env.goal_radius**2:
                    terminated = True
                    reward = 1000
            
            reward_miniepisode+=reward
            
            self.env.t += 1
            truncated = False
            if self.env.t>self.env.Horizon:
                truncated = True
            
            if self.env.render_mode == "human":
                self.env.render()
            
            if terminated or truncated:
                break
        
        self.env.observation = np.concatenate((self.env.get_lidar_reading(),self.env.robot_position))
        
        return self.env.observation, reward_miniepisode, terminated, truncated, {}
        


# You can copy-paste the code for the custom callback LoggingAndSavingCallback
# that you wrote for training3.py. All you need to change is the code for
# initiating the environment during testing.
class LoggingAndSavingCallback(BaseCallback):
    def __init__(self, test_period, test_count, verbose=0):
        super().__init__(verbose)
        self.test_period = test_period
        self.test_count = test_count
        self.training_rewards = []
        self.best_avg_reward = -np.inf
        self.timesteps = 0
        
    def _on_step(self) -> bool:
        self.timesteps += 1
        # Get reward and done flag for the current step
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        # Track cumulative reward for current episode
        if not hasattr(self, 'current_episode_reward'):
            self.current_episode_reward = 0.0
        self.current_episode_reward += reward
        
        # End of episode: store and save training reward
        if done:
            self.training_rewards.append(self.current_episode_reward)
            np.save('training_log.npy', np.array(self.training_rewards))
            self.current_episode_reward = 0.0
        
        # Test periodically
        if self.timesteps % self.test_period == 0:
            self.model.save("LATEST_MODEL")
            
            # Create new testing environment - use ModifiedRobotNavigationEnv for proper testing
            test_base_env = RobotNavigationEnv(render_mode=None)
            test_env = ModifiedRobotNavigationEnv(test_base_env, H=15)  # Same H as training
            
            test_rewards = []
            for _ in range(self.test_count):
                obs, _ = test_env.reset()
                done = False
                episode_reward = 0.0
                
                while not done:
                    # The model predicts (δₓ, δᵧ) - the offset for intermediate goal
                    action, _ = self.model.predict(obs, deterministic=True)
                    # ModifiedRobotNavigationEnv handles the rest:
                    # - Computes intermediate goal as current_pos + (δₓ, δᵧ)  
                    # - Uses conventional policy to move toward intermediate goal
                    obs, reward, terminated, truncated, _ = test_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                test_rewards.append(episode_reward)
            
            avg_reward = np.mean(test_rewards)
            
            # Log testing reward
            try:
                testing_log = list(np.load("testing_log.npy"))
            except FileNotFoundError:
                testing_log = []
            testing_log.append(avg_reward)
            np.save("testing_log.npy", np.array(testing_log))
            
            # Save best model
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.model.save("BEST_MODEL")
            
            test_env.close()
        
        return True
            
            
# Initiate the robot navigation environment. 
env = RobotNavigationEnv()
H = 15  # Reduced H for more frequent decision making and better learning

def make_env():
    base_env = RobotNavigationEnv()
    wrapped_env = ModifiedRobotNavigationEnv(base_env, H)
    monitored_env = Monitor(wrapped_env)
    return monitored_env

env = DummyVecEnv([make_env for _ in range(4)])


# Initiate an instance of the LoggingAndSavingCallback. 
test_period = 10000   
test_count = 5        # For faster testing
callback = LoggingAndSavingCallback(test_period, test_count)


# The code that you use to train the RL agent for the robot navigation environment
# goes below this line. The total number of lines is unlikely to be more than 10.

# Initialize PPO model with optimized hyperparameters for continuous action space
model = TD3(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,   
    batch_size= 256,        # Size of minibatches 
    policy_kwargs=dict(net_arch=[256, 256])  # Customizing the policy network
)


# Train the model with more timesteps for continuous action space
model.learn(total_timesteps=3000000, callback=callback)


# Close the robot navigation environment.
env.close()


# Write just ONE line of code below to save the model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL4.
model.save('MODEL4')