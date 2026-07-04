import gym
from gym import spaces
import numpy as np

class GymTrafficEnv(gym.Env):
    def __init__(self):
        super(GymTrafficEnv, self).__init__()
        # System parameters
        self.arrival_prob_road1 = 0.28
        self.arrival_prob_road2 = 0.4
        self.departure_prob_green = 0.9
        self.queue_length = 1810
        self.max_time_slots = 1800
        
        # Define action space: 0 = Keep current light, 1 = Switch light (only if allowed)
        self.action_space = spaces.Discrete(2)
        
        # (queue1, queue2, green_road, time_since_last_switch)
        self.observation_space = spaces.MultiDiscrete([self.queue_length + 1,  self.queue_length + 1, 2, 11])
        
        self.reset()
        
    def reset(self):
        # Initialize queues randomly between 0 and 10
        self.queue_1 = np.random.randint(0, 11)
        self.queue_2 = np.random.randint(0, 11)
        self.green_road = np.random.choice([1, 2])  # 1 for road1, 2 for road2
        self.time_since_switch = 10  # Initially allow switch
        self.time_slot = 0
        
        return self._get_state()
        
    def _get_state(self):
        # Return state with queue lengths 
        return np.array([
            self.queue_1,
            self.queue_2,
            self.green_road,  # Convert to 0 or 1
            self.time_since_switch 
        ])
        
    def step(self, action):
        # Apply action - only switch if time since last switch is >= 10
        if action == 1 and self.time_since_switch >= 10:
            self.green_road = 3 - self.green_road  # Switch from 1 to 2 or 2 to 1
            self.time_since_switch = 0
        else:
            self.time_since_switch += 1
            
        # Arrivals
        if np.random.rand() < self.arrival_prob_road1: self.queue_1 += 1
        if np.random.rand() < self.arrival_prob_road2: self.queue_2 += 1
            
        # Departures
        if self.green_road == 1:
            # Road 1 is green
            self._try_departure(1)
            self._try_delayed_departure(2)
        else:
            # Road 2 is green
            self._try_departure(2)
            self._try_delayed_departure(1)
            
        self.time_slot += 1
        
        # Compute reward based on actual queue lengths (not capped)
        reward = -(self.queue_1 + self.queue_2)
        
        # Get state representation 
        state = self._get_state()
        
        terminated = False
        truncated = self.time_slot >= self.max_time_slots
        info = {}
        
        return state, reward, terminated, truncated, info
        
    def _try_departure(self, road):
        """Try departure for a road with green light"""
        if np.random.rand() < self.departure_prob_green:
            if road == 1 and self.queue_1 > 0:
                self.queue_1 -= 1
            elif road == 2 and self.queue_2 > 0:
                self.queue_2 -= 1
                
    def _try_delayed_departure(self, road):
        """Try departure for a road with red light but within 10 seconds of switching"""
        if self.time_since_switch < 10:  # Only applies within 10 time slots after switching
        
            delta = self.time_since_switch
            
            # Calculate departure probability using the formula from the problem
            prob = 0.9 * (1 -  ((delta**2)/100)) 
            
            if np.random.rand() < prob:
                if road == 1 and self.queue_1 > 0: self.queue_1 -= 1
                elif road == 2 and self.queue_2 > 0: self.queue_2 -= 1
                
    def set_state(self, state):
        """
        Forcefully set the environment to a given state: (queue1, queue2, green road, time_since_switch)
        """
        self.queue_1 = state[0]
        self.queue_2 = state[1]
        self.green_road = state[2]
        self.time_since_switch = state[3]
                