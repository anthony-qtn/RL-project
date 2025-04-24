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
from dqn import DQN
from env import create_env
from train import hyperparameters, eval_agent
from save import get_latest_model_path

env = create_env()

def visualize_env_constant_action(env, action=1):
    _, _ = env.reset()
    done = False
    while not done:
        action = action
        obs, reward, terminated, truncated, info = env.step(action)  # Pass an integer, not an array
        done = terminated or truncated
        print("reward:", reward)
        env.render()

def visualize_env_agent(env, agent):
    obs, _ = env.reset()
    done = False
    while not done :
        action = agent.get_optimal_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()

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

model_path = get_latest_model_path()
model_path = 'task1/models/model3.pt'

print("evaluating model at ", model_path)

agent.load_model(path = model_path)
visualize_env_agent(env, agent)
#print(eval_agent(agent, env, n_sim=10))