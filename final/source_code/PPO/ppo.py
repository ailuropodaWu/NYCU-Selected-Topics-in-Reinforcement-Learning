from racecar_gym.env import RaceEnv
from stable_baselines3 import PPO
from stable_baselines3.common import vec_env

if __name__ == "__main__":
    env = RaceEnv(
        scenario='circle_cw_competition_collisionStop', # 'austria_competition'
        render_mode='rgb_array_birds_eye',
        reset_when_collision=False, # Only work for 'austria_competition' and 'austria_competition_collisionStop'
    )
    print(env.action_space)
    obs, info = env.reset()
    terminated = False
    while not terminated:
        action = (1.0, 0.0) # Motor and steering
        obs, rew, terminated, truncated, info = env.step(action)