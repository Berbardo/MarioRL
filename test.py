from dqn import DQN
from runner import evaluate, make_env

if __name__ == "__main__":
    env = make_env()()

    agent = DQN(env.observation_space, env.action_space)
    agent.load_model("trained_models/305rt.pth")

    returns = evaluate(agent, env, 1, True)