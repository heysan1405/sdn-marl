"""
Graph Neural Network encoder for SDN-MARL.

The encoder converts a variable-sized network graph into a
fixed-size graph embedding.

Node features:
    [degree,
     node_load,
     normalized_x,
     normalized_y,
     is_source,
     is_destination,
     is_failed]

Edge features:
    [utilization,
     capacity,
     delay,
     packet_loss,
     is_failed]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """
    Edge-aware Graph Attention Network.

    Supports variable numbers of nodes and edges.

    Input:
        node_features: [N, node_feature_dim]
        edge_index:    [2, E]
        edge_features: [E, edge_feature_dim]

    Output:
        graph_embedding: [B, graph_embedding_dim]
    """

    def __init__(
        self,
        node_feature_dim=7,
        edge_feature_dim=5,
        hidden_dim=64,
        graph_embedding_dim=128,
        num_layers=2,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()

        if node_feature_dim <= 0:
            raise ValueError(
                "node_feature_dim must be positive."
            )

        if edge_feature_dim <= 0:
            raise ValueError(
                "edge_feature_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if graph_embedding_dim <= 0:
            raise ValueError(
                "graph_embedding_dim must be positive."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if heads <= 0:
            raise ValueError(
                "heads must be positive."
            )

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.graph_embedding_dim = graph_embedding_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First GAT layer
        self.convs.append(
            GATConv(
                in_channels=node_feature_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=edge_feature_dim,
            )
        )

        self.norms.append(
            nn.LayerNorm(hidden_dim)
        )

        # Remaining GAT layers
        for _ in range(num_layers - 1):

            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_feature_dim,
                )
            )

            self.norms.append(
                nn.LayerNorm(hidden_dim)
            )

        # Convert node embeddings to graph embedding
        self.output_layer = nn.Linear(
            hidden_dim,
            graph_embedding_dim,
        )

        self.output_norm = nn.LayerNorm(
            graph_embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features,
        edge_index,
        edge_features,
        batch=None,
    ):
        """
        Encode graph.

        Parameters
        ----------
        node_features:
            Tensor [N, node_feature_dim]

        edge_index:
            Tensor [2, E]

        edge_features:
            Tensor [E, edge_feature_dim]

        batch:
            Tensor [N] assigning each node to a graph.
            If None, assumes one graph.

        Returns
        -------
        Tensor:
            [B, graph_embedding_dim]
        """

        if node_features.dim() != 2:
            raise ValueError(
                "node_features must have shape [N, F]."
            )

        if edge_index.dim() != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_features.dim() != 2:
            raise ValueError(
                "edge_features must have shape [E, F]."
            )

        if node_features.shape[1] != self.node_feature_dim:
            raise ValueError(
                f"Expected {self.node_feature_dim} node "
                f"features, got {node_features.shape[1]}."
            )

        if edge_features.shape[1] != self.edge_feature_dim:
            raise ValueError(
                f"Expected {self.edge_feature_dim} edge "
                f"features, got {edge_features.shape[1]}."
            )

        if edge_index.numel() > 0:

            if edge_index.min() < 0:
                raise ValueError(
                    "edge_index contains negative indices."
                )

            if edge_index.max() >= node_features.shape[0]:
                raise ValueError(
                    "edge_index contains invalid node indices."
                )

        if batch is None:

            batch = torch.zeros(
                node_features.shape[0],
                dtype=torch.long,
                device=node_features.device,
            )

        x = node_features

        for conv, norm in zip(
            self.convs,
            self.norms,
        ):

            x = conv(
                x,
                edge_index,
                edge_attr=edge_features,
            )

            x = norm(x)

            x = F.elu(x)

            x = self.dropout(x)

        x = self.output_layer(x)

        x = self.output_norm(x)

        graph_embedding = global_mean_pool(
            x,
            batch,
        )

        return graph_embedding