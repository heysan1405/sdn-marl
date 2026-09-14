import random
import numpy as np
import os

# Optional PyTorch support with seamless NumPy fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class QNetwork(nn.Module):
        """Deep Q-Network for Congestion Penalty Selection."""

        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super(QNetwork, self).__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)


class ReplayBuffer:
    """Experience Replay Memory Buffer."""

    def __init__(self, capacity=5000):
        from collections import deque
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    RL Congestion Agent supporting PyTorch Deep Q-Network (DQN)
    and NumPy-based Tabular Q-Learning fallback.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.95,
        buffer_capacity=5000,
        batch_size=32
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.use_torch = HAS_TORCH

        if self.use_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = QNetwork(state_dim, action_dim).to(self.device)
            self.target_network = QNetwork(state_dim, action_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        else:
            # NumPy Q-table fallback
            print("[DQN Agent] PyTorch not detected. Initializing NumPy Q-Learning engine.")
            self.q_table = {}

        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def _state_to_key(self, state):
        """Discretizes state vector for NumPy Q-table lookup."""
        discretized = np.round(state[:self.action_dim], 1)
        return tuple(discretized)

    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        if self.use_torch:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_t)
            return torch.argmax(q_values, dim=1).item()
        else:
            key = self._state_to_key(state)
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.action_dim)
            return int(np.argmax(self.q_table[key]))

    def train_step(self):
        """Performs one optimization step."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        if self.use_torch:
            states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

            q_values = self.q_network(states_t)
            state_action_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = self.target_network(next_states_t)
                max_next_q_values = next_q_values.max(1)[0]
                expected_state_action_values = rewards_t + (self.gamma * max_next_q_values * (1 - dones_t))

            loss = nn.MSELoss()(state_action_values, expected_state_action_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return loss.item()
        else:
            # NumPy Q-learning update
            for i in range(len(states)):
                s_key = self._state_to_key(states[i])
                ns_key = self._state_to_key(next_states[i])
                a = actions[i]
                r = rewards[i]
                d = dones[i]

                if s_key not in self.q_table:
                    self.q_table[s_key] = np.zeros(self.action_dim)
                if ns_key not in self.q_table:
                    self.q_table[ns_key] = np.zeros(self.action_dim)

                target = r if d else r + self.gamma * np.max(self.q_table[ns_key])
                self.q_table[s_key][a] += 0.1 * (target - self.q_table[s_key][a])

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            return 0.01

    def update_target_network(self):
        if self.use_torch:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        if self.use_torch:
            torch.save(self.q_network.state_dict(), path)
            print(f"[DQN] PyTorch model saved to {path}")
        else:
            np.save(path.replace(".pt", ".npy"), self.q_table)
            print(f"[DQN] NumPy Q-Table saved to {path.replace('.pt', '.npy')}")
