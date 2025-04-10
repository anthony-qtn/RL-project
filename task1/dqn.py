import random

class ReplayBuffer:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.position = 0
        self.memory = []

    def add(self, s, a, r, sprime, aprime):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (s, a, r, sprime, aprime)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)
    
    def __len__(self):
        return len(self.memory)