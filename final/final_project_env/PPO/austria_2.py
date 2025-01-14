import torch
from ppo_agent import PPOAgent
# use full motor range and discretize_num 100
weight = '/home/hcis-s17/multimodal_manipulation/patrick/RL/final/final_project_env/PPO/log/austria_original_reward_2/model_22442664_817.pth'
config = {
	"gpu": True,
	"training_steps": 1e8,
	"update_sample_count": 10000,
	"discount_factor_gamma": 0.99,
	"discount_factor_lambda": 0.95,
	"clip_epsilon": 0.2,
	"max_gradient_norm": 0.5,
	"batch_size": 128,
	"logdir": 'log/austria_original_reward_2/',
	"update_ppo_epoch": 3,
	"learning_rate": 2.5e-4,
	"value_coefficient": 0.5,
	"entropy_coefficient": 0.01,
	"horizon": 128,
	"env_id": 'ALE/Enduro-v5',
	"eval_interval": 100,
	"eval_episode": 3,
	"scenario": 'austria_competition',
	"motor_range": [-1., 1.],
	"steering_range": [-1., 1.],
	"discretize_num": 100
}
if __name__ == '__main__':
	agent = PPOAgent(config)
	agent.net.load_state_dict(torch.load('/home/hcis-s17/multimodal_manipulation/patrick/RL/final/final_project_env/PPO/log/austria_original_reward_2/model_4992581_769.pth', weights_only=True))
	agent.train()



