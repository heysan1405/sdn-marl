"""
DQN Congestion Agent for SDN-MARL.
"""

import copy
import random

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.replay_buffer import ReplayBuffer
from models.gnn_encoder import GNNEncoder
from models.congestion_network import CongestionNetwork


class CongestionAgent:
    """
    Agent responsible for congestion management.

    Actions:
        0 -> Maintain
        1 -> Reroute
        2 -> Rate-limit
    """

    MAINTAIN = 0
    REROUTE = 1
    RATE_LIMIT = 2

    NUM_ACTIONS = 3

    def __init__(
        self,
        node_feature_dim=7,
        edge_feature_dim=5,
        demand_feature_dim=1,

        gnn_hidden_dim=64,
        graph_embedding_dim=128,
        gnn_layers=2,
        gnn_heads=4,
        gnn_dropout=0.1,

        hidden_dim=128,

        learning_rate=1e-3,
        gamma=0.99,

        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,

        replay_capacity=100_000,
        batch_size=32,

        target_update_frequency=10,

        device=None,
    ):

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_frequency = (
            target_update_frequency
        )

        self.learn_steps = 0

        # -------------------------------------------------
        # Online networks
        # -------------------------------------------------

        self.online_gnn = GNNEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=gnn_hidden_dim,
            graph_embedding_dim=graph_embedding_dim,
            num_layers=gnn_layers,
            heads=gnn_heads,
            dropout=gnn_dropout,
        ).to(self.device)

        self.online_network = CongestionNetwork(
            graph_embedding_dim=graph_embedding_dim,
            demand_feature_dim=demand_feature_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # -------------------------------------------------
        # Target networks
        # -------------------------------------------------

        self.target_gnn = copy.deepcopy(
            self.online_gnn
        ).to(self.device)

        self.target_network = copy.deepcopy(
            self.online_network
        ).to(self.device)

        self._freeze_target_networks()

        # -------------------------------------------------
        # Optimizer
        # -------------------------------------------------

        self.optimizer = torch.optim.Adam(
            list(self.online_gnn.parameters())
            + list(self.online_network.parameters()),
            lr=learning_rate,
        )

        # -------------------------------------------------
        # Replay buffer
        # -------------------------------------------------

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity
        )

    def _freeze_target_networks(self):

        self.target_gnn.eval()
        self.target_network.eval()

        for parameter in self.target_gnn.parameters():
            parameter.requires_grad = False

        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

    def update_target_network(self):

        self.target_gnn.load_state_dict(
            self.online_gnn.state_dict()
        )

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self._freeze_target_networks()

    def _state_to_tensors(self, state):

        required_keys = {
            "node_features",
            "edge_index",
            "edge_features",
            "demand_features",
        }

        missing = (
            required_keys
            - set(state.keys())
        )

        if missing:
            raise KeyError(
                f"Missing state keys: {sorted(missing)}"
            )

        node_features = torch.as_tensor(
            state["node_features"],
            dtype=torch.float32,
            device=self.device,
        )

        edge_index = torch.as_tensor(
            state["edge_index"],
            dtype=torch.long,
            device=self.device,
        )

        edge_features = torch.as_tensor(
            state["edge_features"],
            dtype=torch.float32,
            device=self.device,
        )

        demand_features = torch.as_tensor(
            state["demand_features"],
            dtype=torch.float32,
            device=self.device,
        )

        return (
            node_features,
            edge_index,
            edge_features,
            demand_features,
        )

    def _q_values(
        self,
        state,
        gnn,
        network,
    ):

        (
            node_features,
            edge_index,
            edge_features,
            demand_features,
        ) = self._state_to_tensors(state)

        graph_embedding = gnn(
            node_features,
            edge_index,
            edge_features,
        )

        return network(
            graph_embedding,
            demand_features,
        )

    def select_action(
        self,
        state,
        training=True,
    ):
        """
        Select a congestion action.

        The original training/evaluation mode is restored
        after greedy action selection.
        """

        # Exploration
        if (
            training
            and random.random() < self.epsilon
        ):

            return random.randrange(
                self.NUM_ACTIONS
            )

        # Save current mode
        was_training_gnn = (
            self.online_gnn.training
        )

        was_training_network = (
            self.online_network.training
        )

        # Temporarily switch to evaluation mode
        self.online_gnn.eval()
        self.online_network.eval()

        try:

            with torch.no_grad():

                q_values = self._q_values(
                    state,
                    self.online_gnn,
                    self.online_network,
                )

                action = int(
                    torch.argmax(
                        q_values[0]
                    ).item()
                )

        finally:

            # Restore previous modes
            self.online_gnn.train(
                was_training_gnn
            )

            self.online_network.train(
                was_training_network
            )

        return action

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        if not 0 <= action < self.NUM_ACTIONS:
            raise ValueError(
                f"Invalid congestion action: {action}"
            )

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def learn(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(
            self.batch_size
        )

        self.online_gnn.train()
        self.online_network.train()

        losses = []

        for (
            state,
            action,
            reward,
            next_state,
            done,
        ) in batch:

            # Current Q-value
            q_values = self._q_values(
                state,
                self.online_gnn,
                self.online_network,
            )

            current_q = q_values[
                0,
                action,
            ]

            # Target Q-value
            with torch.no_grad():

                if done:

                    target_q = torch.tensor(
                        reward,
                        dtype=torch.float32,
                        device=self.device,
                    )

                else:

                    next_q_values = (
                        self._q_values(
                            next_state,
                            self.target_gnn,
                            self.target_network,
                        )
                    )

                    next_max_q = (
                        next_q_values[0].max()
                    )

                    target_q = (
                        reward
                        + self.gamma
                        * next_max_q
                    )

            loss = F.smooth_l1_loss(
                current_q,
                target_q,
            )

            losses.append(loss)

        total_loss = torch.stack(
            losses
        ).mean()

        self.optimizer.zero_grad()

        total_loss.backward()

        clip_grad_norm_(
            list(
                self.online_gnn.parameters()
            )
            + list(
                self.online_network.parameters()
            ),
            max_norm=10.0,
        )

        self.optimizer.step()

        self.learn_steps += 1

        if (
            self.learn_steps
            % self.target_update_frequency
            == 0
        ):

            self.update_target_network()

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon
            * self.epsilon_decay,
        )

        return float(
            total_loss.item()
        )

    def save(self, path):

        torch.save(
            {
                "online_gnn":
                    self.online_gnn.state_dict(),

                "online_network":
                    self.online_network.state_dict(),

                "target_gnn":
                    self.target_gnn.state_dict(),

                "target_network":
                    self.target_network.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),

                "epsilon":
                    self.epsilon,

                "learn_steps":
                    self.learn_steps,
            },
            path,
        )

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.online_gnn.load_state_dict(
            checkpoint["online_gnn"]
        )

        self.online_network.load_state_dict(
            checkpoint["online_network"]
        )

        self.target_gnn.load_state_dict(
            checkpoint["target_gnn"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]
        self.learn_steps = checkpoint[
            "learn_steps"
        ]

        self._freeze_target_networks()