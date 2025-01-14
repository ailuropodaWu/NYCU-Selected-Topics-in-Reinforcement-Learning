from collections import deque
import cv2
from matplotlib.pylab import f
from itertools import product
import numpy as np
from racecar_gym.env import RaceEnv

import gymnasium as gym
from numpy import array, float32


class WrappedRaceEnv:
    def __init__(
        self, 
        N_frame=4, 
        scenario='circle_cw_competition_collisionStop', 
        discretize_num=10,
        steering_range=(-1, 1),
        motor_range=(-1, 1),
        test=False
    ):  
        self.env = RaceEnv(
            scenario=scenario,
            render_mode='rgb_array_birds_eye',
            reset_when_collision=False if 'collisionStop' in scenario else True,
        )
        print(scenario)
        print(self.env.env.reset_when_collision)

        if type(discretize_num) == int:
            discretize_num = [discretize_num] * 2
        action_motor = np.linspace(motor_range[0], motor_range[1], discretize_num[0])
        action_steering = np.linspace(steering_range[0], steering_range[1], discretize_num[1])
        self.action_list = np.array(list(product(action_motor, action_steering)))
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.observation_space = self.env.observation_space
        self.ep_len = 0
        self.frames = deque(maxlen=N_frame)
        self.test = test

    def step(self, action):
        action = self.action_list[action]
        obs, reward, terminates, truncates, info = self.env.step(action)
        original_reward = reward
        original_terminates = terminates
        self.ep_len += 1

        obs = self.wrap_obs(obs)

        if self.test:
            # enable this line to recover the original reward
            reward = original_reward
            # enable this line to recover the original terminates signal, disable this to accerlate evaluation
            terminates = original_terminates

        return obs, reward, terminates, truncates, info
    
    def reset(self, seed=None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        self.ep_len = 0
        
        # convert to grayscale obs = 128*128*3
        obs = self.wrap_obs(obs)
        return obs, info
    
    def wrap_obs(self, obs):
        # convert to grayscale obs = 128*128*3
        obs = np.transpose(obs, (1, 2, 0))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        if len(self.frames) != self.frames.maxlen:
            for _ in range(self.frames.maxlen - len(self.frames)):
                self.frames.append(obs)
        else:
            self.frames.append(obs)
        obs = np.stack(self.frames, axis=0)
        return obs 
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

if __name__ == '__main__':
    env = WrappedRaceEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    total_length = 0
    t = 0
    road = 0
    grass = 0
    while not done:
        t += 1
        action = env.action_space.sample()
        # action[2] = 0.0
        obs, reward, terminates, truncates, info = env.step(action)
        road_pixel_count, grass_pixel_count = info['road_pixel_count'], info['grass_pixel_count']
        road = road_pixel_count
        grass = grass_pixel_count
        print(f'{t}: reward: {reward}')
        print(f'road: {road}, grass: {grass}')
        total_reward += reward
        total_length += 1
        # env.render()
        if terminates or truncates:
            done = True

    print("Total reward: ", total_reward)
    print("Total length: ", total_length)
    env.close()