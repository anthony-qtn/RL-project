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

LOGGING = True

hyperparameters = {
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_capacity": 1000,
    "update_target_every": 10,
    "epsilon_start": 0.9,
    "decrease_epsilon_factor": 2000,
    "epsilon_min": 0.05,
    "learning_rate": 1e-1,
    "N_episodes": 1200,
    "hidden_size": 128,
    "eval_every": 10,
}

if LOGGING:
    mlflow.start_run()
    mlflow.log_params(hyperparameters)

config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "vehicles_count": 15,
    "duration": 60,  # [s]
    "initial_spacing": 0,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.5,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 0.1,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,
    "reward_speed_range": [
        20,
        30,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
}

env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.unwrapped.configure(config_dict) # type: ignore
obs, _ = env.reset()

actions = env.action_space
states = env.observation_space

print("Action Space:", actions)
print("Observation Space:", states)

obs, _ = env.reset()

def visualize_env_constant_action(env, steps=100, action=1):
    _, _ = env.reset()
    for _ in range(steps):
        action = action
        obs, reward, done, truncated, info = env.step(action)  # Pass an integer, not an array
        print("reward:", reward)
        env.render()

def visualize_env_agent(env, agent, steps=100):
    obs, _ = env.reset()
    for _ in range(steps):
        action = agent.get_optimal_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

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

def train_agent(agent:DQN, env, N_episodes = hyperparameters["N_episodes"], eval_every = hyperparameters["eval_every"], timeit = True) -> tuple[list[float], list[float]]:
    mean_rewards = []
    losses = []
    t0 = time.time()
    for episode in range(N_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            loss = agent.update(state, action, reward, done, next_state)
            state = next_state
            if loss is not None:
                losses.append(loss)

            done = terminated or truncated

        if episode % eval_every == 0:
            episode_rewards = eval_agent(agent, env, n_sim=5)
            mean_reward = np.mean(episode_rewards)
            if LOGGING:
                mlflow.log_metric("mean_reward", float(mean_reward), step=episode)
                mlflow.log_metric("epsilon", agent.epsilon, step=episode)
            mean_rewards.append(mean_reward)

    if timeit:
        t1 = time.time()
        fulltime = t1 - t0
        time_per_episode = fulltime / N_episodes
        print(f"Time per episode: {time_per_episode}")

    return mean_rewards, losses

mean_rewards, losses = train_agent(agent, env, N_episodes=hyperparameters["N_episodes"], eval_every=hyperparameters["eval_every"])

agent.save_model(path = "task1/models/test1.pt")

if LOGGING:
    mlflow.end_run()