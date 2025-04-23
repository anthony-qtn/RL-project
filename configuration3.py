"""
Groupe :
Pierre JOURDIN
Aymeric CONTI
Anthony QUENTIN
"""

import pickle

from utils import make_env

# Define configuration for merge-v0 using kinematics observations
config_dict = {
    "observation": {
        "type": "Kinematics",  # Use kinematic-based observation
        "features": [
            "x",
            "y",
            "vx",
        ],  # Position and velocity features (vy always equals 0)
        "vehicles_count": 5,  # Number of vehicles to observe
        "as_image": False,  # Return observation as vector (not as image)
        "align_to_vehicle_axes": True,  # Align observation with vehicle's axes
    },
    "action": {
        "type": "DiscreteMetaAction",  # Use discrete meta actions
        "longitudinal": True,  # Enable throttle control
        "lateral": True,  # Enable steering control
    },
    "simulation_frequency": 16,  # Simulation frequency in Hz
    "policy_frequency": 4,  # Policy frequency in Hz
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # Width of the screen for rendering
    "screen_height": 150,  # Height of the screen for rendering
    "centering_position": [0.3, 0.5],  # Agent centering on screen
    "scaling": 5.5,  # Zoom factor for rendering
    "show_trajectories": False,  # Do not show trajectories
    "render_agent": True,  # Render the agent vehicle
    "offscreen_rendering": False,  # Render on screen (not offscreen)
    "lane_change_reward": 0,  # Lane change reward setting
    "vehicle_count": 10,  # Number of vehicles in the simulation
}

# Save the config to a file (optional)
with open("configs/config3.pkl", "wb") as f:
    pickle.dump(config_dict, f)

env = make_env(task_idx=3)

obs, _ = env.reset()
print(obs)
print("Action Space:", env.action_space)
print("Observation Space:", env.observation_space)
