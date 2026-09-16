import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import GNNEncoder, RoutingNetwork

from .replay_buffer import ReplayBuffer


class RoutingAgent:
    """
    DQN routing agent for topology-generalized SDN routing.

    The agent contains:

        Online GNN
        Online Routing Network

        Target GNN
        Target Routing Network

    The agent receives graph-based states with variable numbers
    of nodes, edges, and candidate paths.

    State format:

        {
            "node_features": [N, 6],
            "edge_index": [2, E],
            "edge_features": [E, 4],
            "demand_features": [1, 1],
            "path_features": [1, K, 3],
            "action_mask": [1, K],
        }

    Node features:

        0 -> degree
        1 -> node_load
        2 -> normalized_x
        3 -> normalized_y
        4 -> is_source
        5 -> is_destination

    Edge features:

        0 -> utilization
        1 -> capacity
        2 -> delay
        3 -> packet_loss

    Demand features:

        0 -> traffic_volume

    Path features:

        0 -> hop_count
        1 -> bottleneck_utilization
        2 -> total_delay

    Action:

        Select one candidate path.

    Epsilon-greedy exploration is implemented here,
    not inside the neural network.
    """

    def __init__(
        self,
        node_feature_dim: int = 6,
        edge_feature_dim: int = 4,
        demand_feature_dim: int = 1,
        path_feature_dim: int = 3,
        gnn_hidden_dim: int = 64,
        graph_embedding_dim: int = 128,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        gnn_dropout: float = 0.1,
        routing_hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 100_000,
        batch_size: int = 32,
        target_update_frequency: int = 10,
        device: str | None = None,
    ):
        super().__init__()

        # --------------------------------------------------
        # Validate hyperparameters
        # --------------------------------------------------

        if node_feature_dim <= 0:
            raise ValueError(
                "node_feature_dim must be greater than 0."
            )

        if edge_feature_dim <= 0:
            raise ValueError(
                "edge_feature_dim must be greater than 0."
            )

        if demand_feature_dim <= 0:
            raise ValueError(
                "demand_feature_dim must be greater than 0."
            )

        if path_feature_dim <= 0:
            raise ValueError(
                "path_feature_dim must be greater than 0."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0."
            )

        if not 0.0 <= gamma <= 1.0:
            raise ValueError(
                "gamma must be between 0 and 1."
            )

        if not 0.0 <= epsilon_start <= 1.0:
            raise ValueError(
                "epsilon_start must be between 0 and 1."
            )

        if not 0.0 <= epsilon_end <= 1.0:
            raise ValueError(
                "epsilon_end must be between 0 and 1."
            )

        if epsilon_end > epsilon_start:
            raise ValueError(
                "epsilon_end cannot be greater than "
                "epsilon_start."
            )

        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be in the range (0, 1]."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if target_update_frequency <= 0:
            raise ValueError(
                "target_update_frequency must be greater "
                "than 0."
            )

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        # --------------------------------------------------
        # Save configuration
        # --------------------------------------------------

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.demand_feature_dim = demand_feature_dim
        self.path_feature_dim = path_feature_dim

        self.gnn_hidden_dim = gnn_hidden_dim
        self.graph_embedding_dim = graph_embedding_dim

        self.gnn_layers = gnn_layers
        self.gnn_heads = gnn_heads
        self.gnn_dropout = gnn_dropout

        self.routing_hidden_dim = routing_hidden_dim

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_frequency = (
            target_update_frequency
        )

        self.learn_steps = 0

        # --------------------------------------------------
        # Online GNN
        # --------------------------------------------------

        self.online_gnn = GNNEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=gnn_hidden_dim,
            graph_embedding_dim=graph_embedding_dim,
            num_layers=gnn_layers,
            heads=gnn_heads,
            dropout=gnn_dropout,
        ).to(self.device)

        # --------------------------------------------------
        # Online routing network
        # --------------------------------------------------

        self.online_routing = RoutingNetwork(
            graph_embedding_dim=graph_embedding_dim,
            demand_feature_dim=demand_feature_dim,
            path_feature_dim=path_feature_dim,
            hidden_dim=routing_hidden_dim,
        ).to(self.device)

        # --------------------------------------------------
        # Target networks
        # --------------------------------------------------

        self.target_gnn = copy.deepcopy(
            self.online_gnn
        ).to(self.device)

        self.target_routing = copy.deepcopy(
            self.online_routing
        ).to(self.device)

        # Target networks are not trained directly.
        self.target_gnn.requires_grad_(False)
        self.target_routing.requires_grad_(False)

        self.target_gnn.eval()
        self.target_routing.eval()

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = optim.Adam(
            list(self.online_gnn.parameters())
            + list(self.online_routing.parameters()),
            lr=learning_rate,
        )

        # --------------------------------------------------
        # Replay buffer
        # --------------------------------------------------

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity
        )

    # ======================================================
    # State handling
    # ======================================================

    def _state_to_tensors(self, state):
        """
        Convert a state dictionary to PyTorch tensors.

        Expected state keys:

            node_features
            edge_index
            edge_features
            demand_features
            path_features
            action_mask
        """

        required_keys = {
            "node_features",
            "edge_index",
            "edge_features",
            "demand_features",
            "path_features",
            "action_mask",
        }

        missing_keys = required_keys - set(
            state.keys()
        )

        if missing_keys:
            raise KeyError(
                "State is missing required keys: "
                f"{sorted(missing_keys)}"
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

        path_features = torch.as_tensor(
            state["path_features"],
            dtype=torch.float32,
            device=self.device,
        )

        action_mask = torch.as_tensor(
            state["action_mask"],
            dtype=torch.bool,
            device=self.device,
        )

        # --------------------------------------------------
        # Basic shape validation
        # --------------------------------------------------

        if node_features.dim() != 2:
            raise ValueError(
                "node_features must have shape [N, F]."
            )

        if edge_index.dim() != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_index.size(0) != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_features.dim() != 2:
            raise ValueError(
                "edge_features must have shape [E, F]."
            )

        if edge_features.size(0) != edge_index.size(1):
            raise ValueError(
                "edge_features must contain one row "
                "for every edge."
            )

        if demand_features.dim() != 2:
            raise ValueError(
                "demand_features must have shape [B, F]."
            )

        if path_features.dim() != 3:
            raise ValueError(
                "path_features must have shape [B, K, F]."
            )

        if action_mask.dim() != 2:
            raise ValueError(
                "action_mask must have shape [B, K]."
            )

        if path_features.size(0) != 1:
            raise ValueError(
                "A routing state must contain exactly "
                "one demand."
            )

        if demand_features.size(0) != 1:
            raise ValueError(
                "demand_features must contain exactly "
                "one demand."
            )

        if action_mask.size(0) != 1:
            raise ValueError(
                "action_mask must contain exactly "
                "one demand."
            )

        if path_features.size(1) != action_mask.size(1):
            raise ValueError(
                "path_features and action_mask must "
                "contain the same number of candidate paths."
            )

        return (
            node_features,
            edge_index,
            edge_features,
            demand_features,
            path_features,
            action_mask,
        )

    # ======================================================
    # Action selection
    # ======================================================

    def select_action(
        self,
        state,
        training: bool = True,
    ) -> int:
        """
        Select a candidate routing path.

        During training:

            epsilon probability:
                choose a random valid action.

            1 - epsilon probability:
                choose the highest-Q valid action.

        During evaluation:

            always choose the highest-Q valid action.

        Returns:
            Integer action index.
        """

        (
            node_features,
            edge_index,
            edge_features,
            demand_features,
            path_features,
            action_mask,
        ) = self._state_to_tensors(state)

        valid_actions = torch.nonzero(
            action_mask[0],
            as_tuple=False,
        ).flatten()

        if valid_actions.numel() == 0:
            raise ValueError(
                "No valid actions are available in "
                "the current state."
            )

        # --------------------------------------------------
        # Epsilon-greedy exploration
        # --------------------------------------------------

        if training and random.random() < self.epsilon:

            random_index = random.randrange(
                valid_actions.numel()
            )

            return int(
                valid_actions[random_index].item()
            )

        # --------------------------------------------------
        # Greedy action
        # --------------------------------------------------

        was_training_gnn = (
            self.online_gnn.training
        )

        was_training_routing = (
            self.online_routing.training
        )

        self.online_gnn.eval()
        self.online_routing.eval()

        with torch.no_grad():

            graph_embedding = self.online_gnn(
                node_features,
                edge_index,
                edge_features,
            )

            q_values = self.online_routing(
                graph_embedding,
                demand_features,
                path_features,
                action_mask,
            )

            action = torch.argmax(
                q_values[0]
            )

        # Restore original training mode.
        self.online_gnn.train(
            was_training_gnn
        )

        self.online_routing.train(
            was_training_routing
        )

        return int(action.item())

    # ======================================================
    # Replay memory
    # ======================================================

    def remember(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ):
        """
        Store one transition in replay memory.
        """

        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    # ======================================================
    # Learning
    # ======================================================

    def learn(self):
        """
        Perform one DQN learning step.

        Because graph sizes vary between experiences,
        transitions are currently processed individually.

        This is intentionally simple and robust for the
        mid-review milestone.

        Returns:
            Mean Huber loss as a float.

            Returns None when the replay buffer does not
            yet contain enough experiences.
        """

        if len(self.replay_buffer) < self.batch_size:
            return None

        experiences = self.replay_buffer.sample(
            self.batch_size
        )

        self.online_gnn.train()
        self.online_routing.train()

        losses = []

        # --------------------------------------------------
        # Process each graph separately
        # --------------------------------------------------

        for (
            state,
            action,
            reward,
            next_state,
            done,
        ) in experiences:

            (
                node_features,
                edge_index,
                edge_features,
                demand_features,
                path_features,
                action_mask,
            ) = self._state_to_tensors(state)

            (
                next_node_features,
                next_edge_index,
                next_edge_features,
                next_demand_features,
                next_path_features,
                next_action_mask,
            ) = self._state_to_tensors(
                next_state
            )

            # --------------------------------------------------
            # Current Q-value
            # --------------------------------------------------

            graph_embedding = self.online_gnn(
                node_features,
                edge_index,
                edge_features,
            )

            q_values = self.online_routing(
                graph_embedding,
                demand_features,
                path_features,
                action_mask,
            )

            if action < 0 or action >= q_values.size(1):
                raise ValueError(
                    f"Invalid action {action} for "
                    f"{q_values.size(1)} candidate paths."
                )

            if not action_mask[0, action]:
                raise ValueError(
                    f"Action {action} is masked as invalid."
                )

            current_q = q_values[0, action]

            # --------------------------------------------------
            # Target Q-value
            # --------------------------------------------------

            with torch.no_grad():

                next_graph_embedding = (
                    self.target_gnn(
                        next_node_features,
                        next_edge_index,
                        next_edge_features,
                    )
                )

                next_q_values = (
                    self.target_routing(
                        next_graph_embedding,
                        next_demand_features,
                        next_path_features,
                        next_action_mask,
                    )
                )

                # Use the action mask directly.
                #
                # If there are no valid next actions,
                # the bootstrap value is zero.

                valid_next_actions = (
                    next_action_mask[0].bool()
                )

                if valid_next_actions.any():

                    next_max_q = next_q_values[
                        0
                    ][valid_next_actions].max()

                else:

                    next_max_q = torch.tensor(
                        0.0,
                        dtype=torch.float32,
                        device=self.device,
                    )

                reward_tensor = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=self.device,
                )

                if done:

                    target_q = reward_tensor

                else:

                    target_q = (
                        reward_tensor
                        + self.gamma * next_max_q
                    )

            # --------------------------------------------------
            # Huber loss
            # --------------------------------------------------

            loss = F.smooth_l1_loss(
                current_q,
                target_q,
            )

            losses.append(loss)

        # --------------------------------------------------
        # Average loss over experiences
        # --------------------------------------------------

        loss = torch.stack(
            losses
        ).mean()

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent excessively large gradients.
        torch.nn.utils.clip_grad_norm_(
            list(self.online_gnn.parameters())
            + list(self.online_routing.parameters()),
            max_norm=10.0,
        )

        self.optimizer.step()

        # --------------------------------------------------
        # Learning step counter
        # --------------------------------------------------

        self.learn_steps += 1

        # --------------------------------------------------
        # Target network update
        # --------------------------------------------------

        if (
            self.learn_steps
            % self.target_update_frequency
            == 0
        ):
            self.update_target_network()

        # --------------------------------------------------
        # Epsilon decay
        # --------------------------------------------------

        self.decay_epsilon()

        return float(
            loss.detach().cpu().item()
        )

    # ======================================================
    # Target network
    # ======================================================

    def update_target_network(self):
        """
        Copy online networks into target networks.
        """

        self.target_gnn.load_state_dict(
            self.online_gnn.state_dict()
        )

        self.target_routing.load_state_dict(
            self.online_routing.state_dict()
        )

        self.target_gnn.eval()
        self.target_routing.eval()

    # ======================================================
    # Epsilon
    # ======================================================

    def decay_epsilon(self):
        """
        Decay exploration rate after a learning step.
        """

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )

    # ======================================================
    # Save
    # ======================================================

    def save(self, path):
        """
        Save the complete agent state.

        Includes:

            online GNN
            online routing network
            target GNN
            target routing network
            optimizer
            epsilon
            learning step counter
        """

        checkpoint = {
            "online_gnn": (
                self.online_gnn.state_dict()
            ),

            "online_routing": (
                self.online_routing.state_dict()
            ),

            "target_gnn": (
                self.target_gnn.state_dict()
            ),

            "target_routing": (
                self.target_routing.state_dict()
            ),

            "optimizer": (
                self.optimizer.state_dict()
            ),

            "epsilon": self.epsilon,

            "learn_steps": self.learn_steps,
        }

        torch.save(
            checkpoint,
            path,
        )

    # ======================================================
    # Load
    # ======================================================

    def load(self, path):
        """
        Load a previously saved agent state.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.online_gnn.load_state_dict(
            checkpoint["online_gnn"]
        )

        self.online_routing.load_state_dict(
            checkpoint["online_routing"]
        )

        self.target_gnn.load_state_dict(
            checkpoint["target_gnn"]
        )

        self.target_routing.load_state_dict(
            checkpoint["target_routing"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = float(
            checkpoint["epsilon"]
        )

        self.learn_steps = int(
            checkpoint["learn_steps"]
        )

        self.target_gnn.eval()
        self.target_routing.eval()

    # ======================================================
    # Utility
    # ======================================================

    def train_mode(self):
        """
        Put online networks into training mode.
        """

        self.online_gnn.train()
        self.online_routing.train()

    def eval_mode(self):
        """
        Put all networks into evaluation mode.
        """

        self.online_gnn.eval()
        self.online_routing.eval()
        self.target_gnn.eval()
        self.target_routing.eval()