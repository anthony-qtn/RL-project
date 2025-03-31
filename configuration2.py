"""
Groupe :
Pierre JOURDIN
Aymeric CONTI
Anthony QUENTIN
"""

import pickle
import gymnasium as gym


config_dict = {
    "observation": {
        "type": "Kinematics",  # Set to Kinematics to use kinematic-based observation
        "features": ['x', 'y', 'vx', 'vy'],  # Use only position and velocity features
        "as_image": False,  # If you prefer the observation as a vector, not an image
        "align_to_vehicle_axes": True  # Align observation to vehicle's local axes
    },
    "action": {
        "type": "ContinuousAction",  # Enable continuous action space
        "longitudinal": True,  # Enable longitudinal control (acceleration)
        "lateral": True        # Enable lateral control (steering)
    },
    "simulation_frequency": 16,  # Hz
    "policy_frequency": 4,  # Hz
    "duration": 60,  # Duration of the simulation in seconds
    "collision_reward": -1,  # Reward for collision
    "lane_centering_cost": 4,  # Cost for not staying centered in the lane
    "action_reward": -0.3,  # Penalty for each action taken
    "controlled_vehicles": 1,  # Number of vehicles controlled by the agent
    "other_vehicles": 1,  # Number of other vehicles
    "screen_width": 600,  # Width of the screen for visualization
    "screen_height": 600,  # Height of the screen for visualization
    "centering_position": [0.5, 0.5],  # Position of the vehicle in the screen
    "scaling": 7,  # Scaling factor for rendering
    "show_trajectories": False,  # Whether to show trajectories of vehicles
    "render_agent": True,  # Whether to render the agent's vehicle
    "offscreen_rendering": False  # Whether to render offscreen
}

# Save the config to a file
with open("config.pkl", "wb") as f:
    pickle.dump(config_dict, f)

# Create and configure the environment
env = gym.make("racetrack-v0", render_mode="rgb_array")
env.unwrapped.configure(config_dict)

# Reset the environment to apply the config
env.reset()

# Check the action and observation spaces
actions = env.action_space
states = env.observation_space

# Print action and observation spaces
print("Action Space:", actions)
print("Observation Space:", states)