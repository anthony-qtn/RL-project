"""
Groupe :
Pierre JOURDIN
Aymeric CONTI
Anthony QUENTIN
"""

import pickle
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from copy import deepcopy
from dqn import DQN
import time
from env import create_env
from save import get_next_model_path

LOGGING = True

hyperparameters = {
    "gamma": 0.95,
    "batch_size": 32,
    "buffer_capacity": 15000,
    "update_target_every": 50,
    "epsilon_start": 0.9,
    "decrease_epsilon_factor": 4000,
    "epsilon_min": 0.05,
    "learning_rate": 5e-4,
    "N_episodes": 4000,
    "hidden_size": 128,
    "eval_every": 10,
}

def eval_agent(agent: DQN, env, n_sim=5) -> np.ndarray:
    """
    ** Solution **
    
    Monte Carlo evaluation of DQN agent.

    Repeat n_sim times:
        * Run the DQN policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    for i in range(n_sim):
        state, _ = env_copy.reset()
        reward_sum = 0
        done = False
        while not done: 
            action = agent.get_action(state, 0)
            state, reward, terminated, truncated, _ = env_copy.step(action)
            reward_sum += reward
            done = terminated or truncated
        episode_rewards[i] = reward_sum
    return episode_rewards

def train_agent(agent:DQN, env, N_episodes = hyperparameters["N_episodes"], eval_every = hyperparameters["eval_every"], timeit = True) -> tuple[list[float], list[float]]:
    mean_rewards = []
    losses = []
    t0 = time.time()
    for episode in range(N_episodes):
        state, _ = env.reset()
        done = False
        episode_loss = 0
        example = (0, 0, 0, 0, 0)
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            loss = agent.update(state, action, reward, done, next_state)
            state = next_state
            if loss is not None:
                episode_loss += loss

            done = terminated or truncated

        if episode % eval_every == 0:
            episode_rewards = eval_agent(agent, env, n_sim=5)
            mean_reward = np.mean(episode_rewards)
            if LOGGING:
                mlflow.log_metric("mean_reward", float(mean_reward), step=episode)
                mlflow.log_metric("epsilon", agent.epsilon, step=episode)
                mlflow.log_metric("loss", float(episode_loss), step=episode)
            mean_rewards.append(mean_reward)
            losses.append(episode_loss)


    if timeit:
        t1 = time.time()
        fulltime = t1 - t0
        time_per_episode = fulltime / N_episodes
        print(f"Time per episode: {time_per_episode}")

    return mean_rewards, losses

if __name__ == "__main__":

    if LOGGING:
        mlflow.start_run()
        mlflow.set_tag("step_represents", "episode")
        mlflow.log_params(hyperparameters)

    env = create_env()

    agent = DQN(action_space = env.action_space,
             observation_space = env.observation_space,
             gamma = hyperparameters["gamma"],
             batch_size = hyperparameters["batch_size"],
             buffer_capacity = hyperparameters["buffer_capacity"],
             update_target_every = hyperparameters["update_target_every"],
             epsilon_start = hyperparameters["epsilon_start"],
             decrease_epsilon_factor = hyperparameters["decrease_epsilon_factor"],
             epsilon_min = hyperparameters["epsilon_min"],
             learning_rate = hyperparameters["learning_rate"],
             hidden_size = hyperparameters["hidden_size"],
             )
    
    get_next_model_path = get_next_model_path()

    print("Training agent, to be saved in ", get_next_model_path)

    mean_rewards, losses = train_agent(agent, env, N_episodes=hyperparameters["N_episodes"], eval_every=hyperparameters["eval_every"])

    agent.save_model(path = get_next_model_path)

    if LOGGING:
        mlflow.end_run()