"""
Groupe :
Pierre JOURDIN
Aymeric CONTI
Anthony QUENTIN
"""

<<<<<<<< HEAD:task1.py
import mlflow
from utils import make_env

LOGGING = False

hyperparameters = {
    "epsilon": 0.05,
}

if LOGGING:
    mlflow.start_run()
    mlflow.log_params(hyperparameters)
========
import pickle
import gymnasium as gym
import highway_env  # noqa: F401
import matplotlib.pyplot as plt
>>>>>>>> 6e81a89156656ef37172a1a4e961defd0255beb9:task1/env.py

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

<<<<<<<< HEAD:task1.py
env = make_env(task_idx=1)
obs, _ = env.reset()
========
def create_env(config_dict = config_dict) -> gym.Env:
>>>>>>>> 6e81a89156656ef37172a1a4e961defd0255beb9:task1/env.py

    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_dict) # type: ignore
    obs, _ = env.reset()

    actions = env.action_space
    states = env.observation_space

    print("Action Space:", actions)
    print("Observation Space:", states)

<<<<<<<< HEAD:task1.py
obs, _ = env.reset()


def visualize_env(env, steps=100, action=1):
    for _ in range(steps):
        action = action
        obs, reward, done, truncated, info = env.step(
            action
        )  # Pass an integer, not an array
        env.render()


visualize_env(env)
========
    return env
>>>>>>>> 6e81a89156656ef37172a1a4e961defd0255beb9:task1/env.py
