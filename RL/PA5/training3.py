import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

class LoggingAndSavingCallback(BaseCallback):
    def __init__(self, test_period, test_count, verbose=0):
        super().__init__(verbose)
        self.test_period = test_period
        self.test_count = test_count
        self.training_rewards = []
        self.current_episode_reward = 0.0
        self.step_counter = 0
        self.best_avg_reward = -np.inf

    def _on_step(self) -> bool:
        #reward = np.clip(self.locals['rewards'][0], -1000, 1000)
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        self.current_episode_reward += reward
        self.step_counter += 1

        if done:
            self.training_rewards.append(self.current_episode_reward)
            np.save("training_log.npy", self.training_rewards)
            self.current_episode_reward = 0.0

        if self.step_counter % self.test_period == 0:
            self.model.save("LATEST_MODEL")
            test_rewards = []

            for _ in range(self.test_count):
                test_env = RobotNavigationEnv()
                obs, _ = test_env.reset()
                terminated, truncated = False, False
                episode_reward = 0.0

                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = test_env.step(action)
                    episode_reward += reward

                test_env.close()
                test_rewards.append(episode_reward)

            avg_reward = np.mean(test_rewards)

            try:
                testing_log = list(np.load("testing_log.npy"))
            except FileNotFoundError:
                testing_log = []

            testing_log.append(avg_reward)
            np.save("testing_log.npy", testing_log)

            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.model.save("BEST_MODEL")

        return True


env = RobotNavigationEnv()

# Setup callback
test_period = 20000
test_count = 10
callback = LoggingAndSavingCallback(test_period, test_count)


model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=5e-5,               
    buffer_size=200_000,             
    learning_starts=5000,             
    batch_size=32,
    tau=0.05,                         
    gamma=0.99,
    train_freq=1,                   
    target_update_interval=100,      
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,       
    exploration_fraction=0.2,         
    policy_kwargs=dict(net_arch=[128, 128]),
    tensorboard_log="./dqn_robotnav_tensorboard/"
)



model.learn(total_timesteps=2_000_000, callback=callback)


model.save("MODEL3")
