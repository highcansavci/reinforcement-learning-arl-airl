import gym
import numpy as np
from agent.arl_airl_agent import Agent


if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    env = gym.wrappers.RecordVideo(env, 'video')
    num_horizon = 20
    batch_size = 5
    n_epochs = 4
    alpha = 3e-4
    agent_ = Agent(input_dim=env.observation_space.shape[0], env=env)
    agent_.load_models()
    n_games = 5

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent_.choose_action(observation)
            observation_, reward, done, info = env.step(np.squeeze(action))
            env.render()
            score += reward
            observation = observation_