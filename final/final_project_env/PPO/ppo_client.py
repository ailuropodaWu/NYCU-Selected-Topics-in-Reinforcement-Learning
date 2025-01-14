import argparse
import json
import numpy as np
import requests
import torch
from ppo_agent import PPOAgent
from austria_2 import config as config_a
from austria_2 import weight as weight_a
from circle import config as config_c
from circle import weight as weight_c


def connect(agent, url: str = 'http://localhost:5000'):
    test_env = agent.test_env
    while True:
        # Get the observation
        response = requests.get(f'{url}')
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break
        obs = json.loads(response.text)['observation']
        obs = np.array(obs).astype(np.uint8)
        obs = test_env.wrap_obs(obs)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = test_env.action_list[agent.act(obs)].squeeze()  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f'{url}', json={'action': action_to_take.tolist()})
        if json.loads(response.text).get('error'):
            print(json.loads(response.text)['error'])
            break

        result = json.loads(response.text)
        terminal = result['terminal']

        if terminal:
            print('Episode finished.')
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='http://localhost:5000', help='The url of the server.')
    parser.add_argument('-c', '--competition', type=str, default='c')
    args = parser.parse_args()


    class RandomAgent:
        def __init__(self, action_space):
            self.action_space = action_space

        def act(self, observation):
            return self.action_space.sample()


    # Initialize the RL Agent
    import gymnasium as gym

    rand_agent = RandomAgent(
        action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))

    if args.competition == 'c':
        config_c['logdir'] = 'log/Eval'
        ppo_agent = PPOAgent(config_c)
        ppo_agent.load(weight_c)
    
    elif args.competition == 'a':
        config_a['logdir'] = 'log/Eval'
        ppo_agent = PPOAgent(config_a)
        ppo_agent.load(weight_a)
    connect(ppo_agent, url=args.url)
