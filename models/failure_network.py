"""
Neural decision network for the Failure Agent.
"""

import torch
import torch.nn as nn


class FailureNetwork(nn.Module):
    """
    Predict Q-values for failure-response actions.

    Actions:
        0 -> No action
        1 -> Reroute
        2 -> Isolate
    """

    NUM_ACTIONS = 3

    def __init__(
        self,
        graph_embedding_dim=128,
        demand_feature_dim=1,
        hidden_dim=128,
    ):
        super().__init__()

        self.demand_encoder = nn.Sequential(
            nn.Linear(
                demand_feature_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        self.network = nn.Sequential(
            nn.Linear(
                graph_embedding_dim + hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ELU(),

            nn.Linear(
                hidden_dim,
                self.NUM_ACTIONS,
            ),
        )

    def forward(
        self,
        graph_embedding,
        demand_features,
    ):
        """
        Returns Q-values for the three failure actions.
        """

        if graph_embedding.dim() != 2:
            raise ValueError(
                "graph_embedding must be 2-dimensional."
            )

        if demand_features.dim() != 2:
            raise ValueError(
                "demand_features must be 2-dimensional."
            )

        demand_embedding = self.demand_encoder(
            demand_features
        )

        combined = torch.cat(
            [
                graph_embedding,
                demand_embedding,
            ],
            dim=-1,
        )

        return self.network(combined)