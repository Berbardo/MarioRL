import time
import math
import numpy as np

from gym.wrappers import Monitor
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from utils.atari_wrappers import wrap_mario

def obs_reshape(obs):
    obs = np.swapaxes(obs, -3, -1)
    obs = np.swapaxes(obs, -1, -2)

    return obs / 255.

def make_env():
    def _make():
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = wrap_mario(env)
        return env
    return _make

def evaluate(agent, env, n_episodes=5, render=False, record=False):
    total_rewards = []

    if record:
        env = Monitor(env, './videos/', force=True)

    for episode in range(n_episodes):

        obs = env.reset()
        obs = obs_reshape(obs)
        total_reward = 0.0
        episode_length = 0

        done = False
        while not done:
            action = agent.act(obs.reshape(1, *obs.shape))
            next_obs, reward, done, _ = env.step(action[0])
            next_obs = obs_reshape(next_obs)
            obs = next_obs
            
            total_reward += reward
            episode_length += 1

            if render:
                env.render()                
                
        total_rewards.append(total_reward)
        
#         print(f">> episode = {episode + 1} / {n_episodes}, total_reward = {total_reward:10.4f}, episode_length = {episode_length}")
        
    if render:
        env.close()

    return np.mean(total_rewards)

def train(agent, env, total_timesteps, break_condition):
    eval_env = make_env()()

    total_rewards = [[] for _ in range(env.num_envs)]
    avg_total_rewards = []

    max_return = 0
    max_eval_return = 0
    total_reward = np.zeros(env.num_envs)
    observations = env.reset()
    observations = obs_reshape(observations)
    timestep = 0
    episode = 0

    t = 0

    start_time = time.time()

    while timestep < total_timesteps:
        actions = agent.act(observations)
        next_observations, rewards, dones, _ = env.step(actions)
        next_observations = obs_reshape(next_observations)
        agent.remember(observations, actions, rewards, next_observations, dones)
        agent.train()
        
        timestep += len(observations)
        t += 1

        total_reward += rewards

        for i in range(env.num_envs):
            if dones[i]:
                total_rewards[i].append((t, timestep, total_reward[i]))
                episode += 1

        if any(G for G in total_rewards):
            episode_returns = sorted(
                list(np.concatenate([G for G in total_rewards if G])),
                key=lambda x: x[1]
            )

            avg_total_rewards.append(np.mean([G[-1] for G in episode_returns[-20:]]))

        total_reward *= 1 - dones
        observations = next_observations

        ratio = math.ceil(100 * timestep / total_timesteps)
        uptime = math.ceil(time.time() - start_time)

        avg_return = avg_total_rewards[-1] if avg_total_rewards else np.nan

        if avg_total_rewards:
            if avg_return > max_return:
                agent.save_model("modelmax.pth")
                max_return = avg_return

        if timestep % 50000 == 0:
            agent.epsilon = 0.1

        if timestep % 10000 == 0:
            agent.save_model("modelper.pth")
        
        if timestep % 10000 == 8:
            eval_return = evaluate(agent, eval_env, 5, False)
            if eval_return >= max_eval_return:
                max_eval_return = eval_return
                agent.save_model("modeleval.pth")
                if max_eval_return > break_condition:
                    print("\n")
                    return avg_total_rewards

        print(f"[{ratio:3d}% / {uptime:3d}s] timestep = {timestep}/{total_timesteps}, episode = {episode:3d}, avg_return = {avg_return:10.4f}, max_avg_return = {max_return:10.4f}, max_eval_return = {max_eval_return:10.4f}\r", end="")

    print("\n")
    return avg_total_rewards