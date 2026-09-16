from collections import deque
import random


class ReplayBuffer:
    """
    Experience replay buffer for graph-based DQN.

    Each experience is stored as:

        (
            state,
            action,
            reward,
            next_state,
            done
        )

    States are stored as dictionaries because graph sizes
    can vary between experiences.
    """

    def __init__(self, capacity: int = 100_000):
        if capacity <= 0:
            raise ValueError(
                "capacity must be greater than 0."
            )

        self.buffer = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ):
        """
        Add one experience to the replay buffer.
        """

        self.buffer.append(
            (
                state,
                int(action),
                float(reward),
                next_state,
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        """
        Randomly sample experiences.

        Raises:
            ValueError:
                If the buffer contains fewer experiences
                than requested.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences. "
                f"Current size: {len(self.buffer)}, "
                f"requested: {batch_size}."
            )

        return random.sample(
            self.buffer,
            batch_size,
        )

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """
        Remove all experiences.
        """

        self.buffer.clear()