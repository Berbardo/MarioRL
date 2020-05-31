from gym.vector import AsyncVectorEnv

from agents.dqn.dqn import DQN
from agents.ppo.ppo import PPO
from utils.runner import train, make_env

if __name__ == "__main__":
    env_fns = [make_env() for _ in range(8)]
    env = AsyncVectorEnv(env_fns)

    agent = PPO(env.single_observation_space, env.single_action_space)

    returns = train(agent, env, 3000000, 500)