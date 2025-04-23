import highway_env  # noqa: F401
import pickle
import gymnasium as gym


def make_env(*, task_idx: int) -> gym.Env:
    names = ["highway-fast-v0", "racetrack-v0", "merge-v0"]
    assert 1 <= task_idx <= 3, f"task_idx should be between 1 and 3 not {task_idx}"

    env = gym.make(names[task_idx - 1], render_mode="rgb_array")
    with open(f"configs/config{task_idx}.pkl", "rb") as f:
        config_dict = pickle.load(f)
        env.unwrapped.configure(config_dict)  # type: ignore
    env.reset()
    return env


def visualize_random(env: gym.Env, steps: int = 100):
    env.reset()
    for _ in range(steps):
        env.step(env.action_space.sample())
        env.render()
    env.close()


def visualize_env_agent(env: gym.Env, agent):
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.get_optimal_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
