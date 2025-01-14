import torch
from ppo_agent import PPOAgent

# larger steer and motor (to make it faster)
config = {
	"gpu": True,
	"training_steps": 1e8,
	"update_sample_count": 10000,
	"discount_factor_gamma": 0.99,
	"discount_factor_lambda": 0.95,
	"clip_epsilon": 0.2,
	"max_gradient_norm": 0.5,
	"batch_size": 128,
	"logdir": 'log/circle_origin_reward_1/',
	"update_ppo_epoch": 3,
	"learning_rate": 2.5e-4,
	"value_coefficient": 0.5,
	"entropy_coefficient": 0.01,
	"horizon": 128,
	"env_id": 'ALE/Enduro-v5',
	"eval_interval": 100,
	"eval_episode": 3,
	"scenario": 'circle_cw_competition_collisionStop', # 'austria_competition'
	"motor_range": [0.5, 1],
	"steering_range": [0.5, 1],
	"discretize_num": 20
}
if __name__ == '__main__':
	agent = PPOAgent(config)
	agent.train()



