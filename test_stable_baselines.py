import gym
import grid2op
from grid2op.gym_compat import GymEnv

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC

env = gym.make('CartPole-v1')
glop_env = grid2op.make("educ_case14_redisp", test=True)
gym_env_raw = GymEnv(glop_env)
model = SAC(MlpPolicy, gym_env_raw, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
model.save("sac_pendulum")


obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()