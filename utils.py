"""
Groupe :
Pierre JOURDIN
Aymeric CONTI
Anthony QUENTIN
"""

import highway_env  # noqa: F401
import os
import pickle
import gymnasium as gym


def make_env(*, task_idx: int, **kwargs) -> gym.Env:
    names = ["highway-fast-v0", "racetrack-v0", "merge-v0"]
    assert 1 <= task_idx <= 3, f"task_idx should be between 1 and 3 not {task_idx}"

    env = gym.make(names[task_idx - 1], render_mode="rgb_array", **kwargs)

    assert os.path.exists(f"configs/config{task_idx}.pkl"), (
        f"Config file for task {task_idx} not found. You should add your config inside the configs folder "
        f"with the name config{task_idx}.pkl"
    )

    with open(f"configs/config{task_idx}.pkl", "rb") as f:
        config_dict = pickle.load(f)
        env.unwrapped.configure(config_dict)  # type: ignore
    env.reset()
    return env


def visualize_random(steps: int = 100, *, task_idx: int = 1):
    env = make_env(task_idx=task_idx)
    for _ in range(steps):
        env.step(env.action_space.sample())
        env.render()
    env.close()


def visualize(
    agent,
    *,
    task_idx: int,
    create_gif: bool = False,
    gif_file: str = "",
    gif_fps: int = 30,
):
    env = make_env(task_idx=task_idx)
    obs, _ = env.reset()
    done = False
    images = []
    while not done:
        action = agent.get_optimal_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        images.append(env.render())
    env.close()

    if create_gif:
        import imageio

        imageio.mimsave(
            gif_file or f"task{task_idx}/simulation.gif", images, fps=gif_fps
        )
