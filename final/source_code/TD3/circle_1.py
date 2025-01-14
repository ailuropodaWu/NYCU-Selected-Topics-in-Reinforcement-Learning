from td3_agent import TD3Agent

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
	"logdir": 'log/circle_1/',
	"update_freq": 2,
	"eval_interval": 10,
	"eval_episode": 10,
	"scenario": "circle_cw_competition_collisionStop",
	"motor_range": [-1, 1],
	"steer_range": [-1, 1],
}
if __name__ == '__main__':
	# my hyperparameters, you can change it as you like
	agent = TD3Agent(config)
	# agent.load('/home/hcis-s17/multimodal_manipulation/patrick/RL/final/final_project_env/TD3/log/circle_1/model_180819_0.pth')
	agent.train()


