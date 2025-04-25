import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.position = 0
        self.memory = []

    def add(self, state, action, reward, done, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, done, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        return random.choices(self.memory, k=batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)
    
class Net(nn.Module):
    """
    Basic neural net.
    """
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class DQN: 
    def __init__(self,
                action_space,
                observation_space,
                gamma,
                batch_size,
                buffer_capacity,
                update_target_every, 
                epsilon_start, 
                decrease_epsilon_factor, 
                epsilon_min,
                learning_rate,
                hidden_size,
                **kwargs
                ): 
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every
        
        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor # larger -> more exploration
        self.epsilon_min = epsilon_min
        
        self.learning_rate = learning_rate

        self.hidden_size = hidden_size
        
        self.reset()
        
    def get_action(self, state: np.ndarray, epsilon=None) -> int | np.int64:
        """
        ** TO BE IMPLEMENTED NOW**

        Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None: 
            epsilon = self.epsilon
            
        # Your code here
        x = random.random()
        if x < epsilon: #random action
            action = self.action_space.sample()
        else:
            q_values = self.get_q(state)
            action = np.argmax(q_values)

        return action
    def get_optimal_action(self, state: np.ndarray) -> int | np.int64:
        return self.get_action(state, 0)
    
    def update(self, state: np.ndarray, action: int | np.int64, reward: float, done: bool, next_state: np.ndarray) -> float:
        """
        ** TO BE COMPLETED **
        """

        # add data to replay buffer
        self.buffer.add(torch.tensor(state, dtype=torch.float32).unsqueeze(0), 
                           torch.tensor([[action]], dtype=torch.int64), 
                           torch.tensor([reward], dtype=torch.float32), 
                           torch.tensor([done], dtype=torch.int64), 
                           torch.tensor(next_state, dtype=torch.float32).unsqueeze(0),
                          )

        if len(self.buffer) < self.batch_size:
            return np.inf

        # get batch
        transitions = self.buffer.sample(self.batch_size)


        state_batch, action_batch, reward_batch, terminated_batch, next_state_batch = tuple(
            [torch.cat(data) for data in zip(*transitions)]
        )

        state_batch = state_batch.view(self.batch_size, -1) # shape (batch_size, obs_size)
        next_state_batch = next_state_batch.view(self.batch_size, -1) # shape (batch_size, obs_size)
        action_batch = action_batch.view(self.batch_size, -1) # shape (batch_size, 1)
        vals = self.q_net.forward(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(next_state_batch).max(1)[0]
            targets = next_state_values * self.gamma + reward_batch
        
        loss = self.loss_function(vals.squeeze(), targets.squeeze())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.n_steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.n_steps += 1

        self.n_eps += 1

        self.decrease_epsilon()

        return loss.detach().numpy()
    
    def get_q(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Q function for a states
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.view(1, -1) # shape (1, obs_size)
        with torch.no_grad():
            output = self.q_net.forward(state_tensor) # shape (1,  n_actions)
        return output.numpy()[0]  # shape  (n_actions)
    
    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (np.exp(-1. * self.n_eps / self.decrease_epsilon_factor ) )
    
    def reset(self):
        hidden_size = self.hidden_size
        
        obs_size = self.observation_space.shape[0]*self.observation_space.shape[1]*self.observation_space.shape[2]
        n_actions = self.action_space.n
        
        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.q_net =  Net(obs_size, hidden_size, n_actions)
        self.target_net = Net(obs_size, hidden_size, n_actions)
        
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)
        
        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0

    def save_model(self, path = "task1/models/dqn.pt"):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path = "task1/models/dqn.pt"):
        self.q_net.load_state_dict(torch.load(path))
    