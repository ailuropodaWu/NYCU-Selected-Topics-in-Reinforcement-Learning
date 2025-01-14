import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent_parallelized import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		def make_env():
			env = gym.make(config["env_id"], render_mode="rgb_array")
			env = gym.wrappers.atari_preprocessing.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1)
			env = gym.wrappers.frame_stack.FrameStack(env, 4)
			return env
		self.env = gym.vector.SyncVectorEnv([make_env for _ in range(self.num_envs)])

  
		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = gym.wrappers.atari_preprocessing.AtariPreprocessing(self.test_env, screen_size=84, grayscale_obs=True, frame_skip=1)
		self.test_env = gym.wrappers.frame_stack.FrameStack(self.test_env, 4)
  
		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.single_action_space.n)
		self.behavior_net.to(self.device)
		self.behavior_net.load_state_dict(torch.load('log/DQN_Parallelized/MsPacman/model_1732265_5230.pth'))
		self.target_net = AtariNetDQN(self.env.single_action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		if random.random() < epsilon:
			action = action_space.sample()
		else:
			observation = torch.from_numpy(np.array(observation)).to(self.device).squeeze(-1)
			observation = observation.unsqueeze(0) if len(observation.shape) == 3 else observation
			action = torch.argmax(self.behavior_net(observation), dim=1).detach().cpu()
		return action
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		for i in range(self.num_envs):
			state, action, reward, next_state, done = self.replay_buffer[i].sample(self.batch_size, self.device)

			### TODO ###
			# calculate the loss and update the behavior network
			# 1. get Q(s,a) from behavior net
			# 2. get max_a Q(s',a) from target net
			# 3. calculate Q_target = r + gamma * max_a Q(s',a)
			# 4. calculate loss between Q(s,a) and Q_target
			# 5. update behavior net

			action = action.type(torch.long)
			q_value = self.behavior_net(state.squeeze(-1)).gather(1, action)
			with torch.no_grad():
				q_next = torch.max(self.target_net(next_state.squeeze(-1)), dim=1, keepdim=True)[0]
				q_target = reward + self.gamma * q_next * (1 - done)
			
			
			criterion = nn.MSELoss()
			loss = criterion(q_value, q_target)

			self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

			self.optim.zero_grad()
			loss.backward()
			self.optim.step()
	
	