"""
Groupe :
Pierre JOURDIN
Aymeric CONTI
Anthony QUENTIN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import wandb

from collections import deque
from tqdm import tqdm

from utils import make_env, visualize


STATE_DIM = 6 * 4  # (x, y, vx, vy, latt_off, ang_off) * 4 cars

# Hyperparameters
GAMMA = 0.8
TAU = 0.001
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
EXPL_NOISE = 0.1
EPOCHS = 2000


class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def push(self, *transition):
        self.buffer.append(tuple(map(np.array, transition)))

    def sample(self):
        batch = random.sample(self.buffer, BATCH_SIZE)
        return [torch.tensor(np.stack(x), dtype=torch.float32) for x in zip(*batch)]

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))

    def get_optimal_action(self, obs, gaussian=True) -> torch.Tensor:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.forward(obs_tensor).detach().numpy()[0]
        if gaussian:
            action += EXPL_NOISE * np.random.randn(2)
        return np.clip(action, -1, 1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x.flatten(start_dim=1), u], dim=1))


def soft_update(target: nn.Module, source: nn.Module):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(TAU * sp.data + (1 - TAU) * tp.data)


def train_one_epoch(
    buffer: ReplayBuffer,
    actor: Actor,
    actor_target: Actor,
    critic: Critic,
    critic_target: Critic,
    actor_opt: optim.Optimizer,
    critic_opt: optim.Optimizer,
):
    state, action, reward, next_state, done = buffer.sample()

    # Critic update
    with torch.no_grad():
        next_q = critic_target(next_state, actor_target(next_state))
        target_q = reward.unsqueeze(1) + GAMMA * (1 - done.unsqueeze(1)) * next_q
    q = critic(state, action)
    critic_loss = nn.MSELoss()(q, target_q)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # Actor update
    actor_loss = -critic(state, actor(state)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    soft_update(actor_target, actor)
    soft_update(critic_target, critic)

    return actor_loss, critic_loss


def train():
    env = make_env(task_idx=2)

    actor = Actor()
    actor_target = Actor()
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic()
    critic_target = Critic()
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    buffer = ReplayBuffer()

    wandb.init(
        project="highway-env",
        config={
            "gamma": GAMMA,
            "tau": TAU,
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "exploration_noise": EXPL_NOISE,
        },
    )

    pbar = tqdm(range(EPOCHS))

    for episode in pbar:
        state = env.reset()[0]
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic_loss = 0

        for _ in range(200):
            action = actor.get_optimal_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += float(reward)

            if len(buffer) > BATCH_SIZE:
                actor_loss, critic_loss = train_one_epoch(
                    buffer,
                    actor,
                    actor_target,
                    critic,
                    critic_target,
                    actor_opt,
                    critic_opt,
                )
                episode_actor_loss += actor_loss.item()
                episode_critic_loss += critic_loss.item()

            if done:
                break

        pbar.set_description(f"Reward: {episode_reward:.2f}")
        wandb.log(
            {
                "episode_reward": episode_reward,
                "actor_loss": episode_actor_loss,
                "critic_loss": episode_critic_loss,
            }
        )

    torch.save(actor.state_dict(), "task2/actor.pth")
    torch.save(critic.state_dict(), "task2/critic.pth")

    wandb.finish()
    env.close()


def test():
    actor = Actor()
    actor.load_state_dict(torch.load("task2/saved/actor.pth"))

    visualize(actor, task_idx=2, create_gif=True)


if __name__ == "__main__":
    # train()
    test()
