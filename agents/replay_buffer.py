"""
Experience replay buffer for DQN agents.
"""

from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity=100_000):

        if capacity <= 0:
            raise ValueError(
                "capacity must be positive."
            )

        self.buffer = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.buffer.append(
            (
                state,
                int(action),
                float(reward),
                next_state,
                bool(done),
            )
        )

    def sample(self, batch_size):

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if len(self.buffer) < batch_size:
            raise ValueError(
                "Not enough experiences."
            )

        return random.sample(
            self.buffer,
            batch_size,
        )

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()