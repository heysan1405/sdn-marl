import torch
import torch.nn as nn


class RoutingNetwork(nn.Module):
    """
    Q-network for routing decisions.

    Each candidate path receives its own Q-value.

    Inputs
    ------
    graph_embedding:
        [batch_size, graph_embedding_dim]

    demand_features:
        [batch_size, demand_feature_dim]

    path_features:
        [batch_size, num_paths, path_feature_dim]

    action_mask:
        [batch_size, num_paths]

        True  = valid candidate path
        False = invalid/padded path

    Output
    ------
    q_values:
        [batch_size, num_paths]
    """

    def __init__(
        self,
        graph_embedding_dim: int,
        demand_feature_dim: int,
        path_feature_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        if graph_embedding_dim <= 0:
            raise ValueError(
                "graph_embedding_dim must be greater than 0"
            )

        if demand_feature_dim <= 0:
            raise ValueError(
                "demand_feature_dim must be greater than 0"
            )

        if path_feature_dim <= 0:
            raise ValueError(
                "path_feature_dim must be greater than 0"
            )

        self.graph_embedding_dim = graph_embedding_dim
        self.demand_feature_dim = demand_feature_dim
        self.path_feature_dim = path_feature_dim
        self.hidden_dim = hidden_dim

        # ---------------------------------------------------------
        # Demand encoder
        # ---------------------------------------------------------
        self.demand_encoder = nn.Sequential(
            nn.Linear(
                demand_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Candidate path encoder
        # ---------------------------------------------------------
        self.path_encoder = nn.Sequential(
            nn.Linear(
                path_feature_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # Q-value network
        # ---------------------------------------------------------
        combined_dim = (
            graph_embedding_dim
            + hidden_dim
            + hidden_dim
        )

        self.q_network = nn.Sequential(
            nn.Linear(
                combined_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        graph_embedding: torch.Tensor,
        demand_features: torch.Tensor,
        path_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # ---------------------------------------------------------
        # Validate graph embedding
        # ---------------------------------------------------------
        if graph_embedding.dim() != 2:
            raise ValueError(
                "graph_embedding must have shape "
                "[batch_size, graph_embedding_dim]"
            )

        batch_size = graph_embedding.size(0)

        if graph_embedding.size(1) != self.graph_embedding_dim:
            raise ValueError(
                f"Expected graph embedding dimension "
                f"{self.graph_embedding_dim}, "
                f"got {graph_embedding.size(1)}"
            )

        # ---------------------------------------------------------
        # Validate demand features
        # ---------------------------------------------------------
        if demand_features.dim() != 2:
            raise ValueError(
                "demand_features must have shape "
                "[batch_size, demand_feature_dim]"
            )

        if demand_features.size(0) != batch_size:
            raise ValueError(
                "graph_embedding and demand_features "
                "must have the same batch size"
            )

        if demand_features.size(1) != self.demand_feature_dim:
            raise ValueError(
                f"Expected {self.demand_feature_dim} demand features, "
                f"got {demand_features.size(1)}"
            )

        # ---------------------------------------------------------
        # Validate path features
        # ---------------------------------------------------------
        if path_features.dim() != 3:
            raise ValueError(
                "path_features must have shape "
                "[batch_size, num_paths, path_feature_dim]"
            )

        if path_features.size(0) != batch_size:
            raise ValueError(
                "graph_embedding and path_features "
                "must have the same batch size"
            )

        if path_features.size(2) != self.path_feature_dim:
            raise ValueError(
                f"Expected {self.path_feature_dim} path features, "
                f"got {path_features.size(2)}"
            )

        num_paths = path_features.size(1)

        if num_paths == 0:
            raise ValueError(
                "At least one candidate path is required"
            )

        # ---------------------------------------------------------
        # Encode demand
        # ---------------------------------------------------------
        demand_embedding = self.demand_encoder(
            demand_features
        )

        # Shape:
        # [B, hidden_dim]
        #
        # Convert to:
        # [B, K, hidden_dim]
        demand_embedding = demand_embedding.unsqueeze(1)

        demand_embedding = demand_embedding.expand(
            -1,
            num_paths,
            -1,
        )

        # ---------------------------------------------------------
        # Encode candidate paths
        # ---------------------------------------------------------
        path_embedding = self.path_encoder(
            path_features
        )

        # Shape:
        # [B, K, hidden_dim]

        # ---------------------------------------------------------
        # Expand graph embedding
        # ---------------------------------------------------------
        graph_embedding = graph_embedding.unsqueeze(1)

        graph_embedding = graph_embedding.expand(
            -1,
            num_paths,
            -1,
        )

        # Shape:
        # [B, K, graph_embedding_dim]

        # ---------------------------------------------------------
        # Combine information
        # ---------------------------------------------------------
        combined = torch.cat(
            [
                graph_embedding,
                demand_embedding,
                path_embedding,
            ],
            dim=-1,
        )

        # Shape:
        # [B, K, combined_dim]

        # ---------------------------------------------------------
        # Calculate Q-value for every path
        # ---------------------------------------------------------
        q_values = self.q_network(
            combined
        ).squeeze(-1)

        # Shape:
        # [B, K]

        # ---------------------------------------------------------
        # Mask invalid actions
        # ---------------------------------------------------------
        if action_mask is not None:

            if action_mask.shape != q_values.shape:
                raise ValueError(
                    "action_mask must have the same shape "
                    "as q_values"
                )

            action_mask = action_mask.bool()

            # Prevent invalid paths from being selected.
            q_values = q_values.masked_fill(
                ~action_mask,
                -torch.inf,
            )

        return q_values

    @staticmethod
    def select_greedy_action(
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select the highest-Q valid candidate path.

        This is ONLY greedy selection.

        Epsilon-greedy exploration belongs in
        routing_agent.py.
        """

        if q_values.dim() != 2:
            raise ValueError(
                "q_values must have shape "
                "[batch_size, num_paths]"
            )

        return torch.argmax(
            q_values,
            dim=-1,
        )