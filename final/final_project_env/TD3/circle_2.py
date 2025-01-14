from td3_agent import TD3Agent

# larger motor and steer range
config = {
	"gpu": True,
	"training_steps": 1e8,
	"gamma": 0.99,
	"tau": 0.005,
	"batch_size": 32,
	"warmup_steps": 1000,
	"total_episode": 100000,
	"lra": 4.5e-5,
	"lrc": 4.5e-5,
	"replay_buffer_capacity": 5000,
	"logdir": 'log/circle_2/',
	"update_freq": 2,
	"eval_interval": 10,
	"eval_episode": 10,
	"scenario": "circle_cw_competition_collisionStop",
	"motor_range": [0.5, 1],
	"steer_range": [0.5, 1],
}
if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	agent = TD3Agent(config)
	# agent.load('/home/hcis-s17/multimodal_manipulation/patrick/RL/final/final_project_env/TD3/log/circle_2/model_175587_0.pth')
	agent.train()


