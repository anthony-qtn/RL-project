import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import VecMonitor

from utils import make_env, visualize

# ========== Paramètres ==========
TASK_IDX = 3  # merge-v0
N_CPUS = 6
BATCH_SIZE = 64
TOTAL_TIMESTEPS = int(2e5)
MODEL_DIR = "task3"
MODEL_PATH = os.path.join(MODEL_DIR, "model")
TENSORBOARD_LOGDIR = MODEL_DIR


# ========== Environnement Vec ==========
def make_vec_env_with_config(n_envs=1, max_episode_steps=60):
    def make_timelimited_env():
        base_env = make_env(task_idx=TASK_IDX)
        return TimeLimit(base_env, max_episode_steps=max_episode_steps)

    # Crée l'environnement vectorisé
    vec_env = SubprocVecEnv([make_timelimited_env for _ in range(n_envs)])

    # Enveloppe avec VecMonitor pour enregistrer les infos de rollout
    return VecMonitor(vec_env, filename=None)


# ========== Entraînement ==========
def train():
    env = make_vec_env_with_config(n_envs=N_CPUS)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=BATCH_SIZE * 12 // N_CPUS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.8,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOGDIR,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modèle sauvegardé à : {MODEL_PATH}")


# ========== Test ==========
def test():
    agent = PPOAgentWrapper(PPO.load(MODEL_PATH))
    visualize(agent, task_idx=TASK_IDX)


class PPOAgentWrapper:
    def __init__(self, model):
        self.model = model

    def get_optimal_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


# ========== Main ==========
if __name__ == "__main__":
    # train()
    for _ in range(5):
        test()

    # agent = PPOAgentWrapper(PPO.load(MODEL_PATH))
    # visualize(agent, task_idx=TASK_IDX, create_gif=True, gif_file="highway_merge_ppo/simulation.gif")
