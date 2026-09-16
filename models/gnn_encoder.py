import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """
    Edge-aware GAT encoder for topology-aware routing.

    The model supports variable-size network topologies.

    Inputs
    ------
    x:
        Node features
        Shape: [num_nodes, node_feature_dim]

    edge_index:
        Graph connectivity
        Shape: [2, num_edges]

    edge_attr:
        Edge/link features
        Shape: [num_edges, edge_feature_dim]

    batch:
        Graph ID for each node.
        Shape: [num_nodes]

        If None, the input is treated as one graph.

    Output
    ------
    graph_embedding:
        Fixed-size graph representation.
        Shape: [num_graphs, graph_embedding_dim]
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        graph_embedding_dim: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if node_feature_dim <= 0:
            raise ValueError(
                "node_feature_dim must be greater than 0"
            )

        if edge_feature_dim <= 0:
            raise ValueError(
                "edge_feature_dim must be greater than 0"
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be greater than 0"
            )

        if graph_embedding_dim <= 0:
            raise ValueError(
                "graph_embedding_dim must be greater than 0"
            )

        if num_layers < 1:
            raise ValueError(
                "num_layers must be at least 1"
            )

        if heads < 1:
            raise ValueError(
                "heads must be at least 1"
            )

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.graph_embedding_dim = graph_embedding_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Keep track of the output dimension of every
        # intermediate GNN layer.
        layer_dimensions = []

        # =========================================================
        # Single-layer GNN
        # =========================================================
        if num_layers == 1:

            self.convs.append(
                GATConv(
                    in_channels=node_feature_dim,
                    out_channels=graph_embedding_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_feature_dim,
                )
            )

        # =========================================================
        # Multiple GNN layers
        # =========================================================
        else:

            # First layer
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

            layer_dimensions.append(hidden_dim)

            # Intermediate layers
            for _ in range(num_layers - 2):

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

                layer_dimensions.append(hidden_dim)

            # Final layer
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=graph_embedding_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_feature_dim,
                )
            )

        # =========================================================
        # Normalization
        # =========================================================
        #
        # There is exactly one normalization layer for every
        # GNN layer except the final one.
        #
        # Example:
        #
        # num_layers = 3
        #
        # GNN 1 -> LayerNorm
        # GNN 2 -> LayerNorm
        # GNN 3 -> no LayerNorm
        #
        for dimension in layer_dimensions:

            self.norms.append(
                nn.LayerNorm(dimension)
            )

        # Final graph-embedding normalization.
        self.output_norm = nn.LayerNorm(
            graph_embedding_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # =========================================================
        # Validate node features
        # =========================================================

        if x.dim() != 2:
            raise ValueError(
                "x must have shape "
                "[num_nodes, node_feature_dim]"
            )

        if x.size(1) != self.node_feature_dim:
            raise ValueError(
                f"Expected {self.node_feature_dim} node features, "
                f"got {x.size(1)}"
            )

        # =========================================================
        # Validate edge index
        # =========================================================

        if edge_index.dim() != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]"
            )

        if edge_index.size(0) != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]"
            )

        if edge_index.size(1) == 0:
            raise ValueError(
                "The graph must contain at least one edge"
            )

        # =========================================================
        # Validate edge features
        # =========================================================

        if edge_attr.dim() != 2:
            raise ValueError(
                "edge_attr must have shape "
                "[num_edges, edge_feature_dim]"
            )

        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError(
                "Number of edge features must match "
                "number of edges"
            )

        if edge_attr.size(1) != self.edge_feature_dim:
            raise ValueError(
                f"Expected {self.edge_feature_dim} edge features, "
                f"got {edge_attr.size(1)}"
            )

        # =========================================================
        # Validate edge indices
        # =========================================================

        if edge_index.min() < 0:
            raise ValueError(
                "edge_index contains a negative node index"
            )

        if edge_index.max() >= x.size(0):
            raise ValueError(
                "edge_index contains a node index outside "
                "the range of x"
            )

        # =========================================================
        # Batch handling
        # =========================================================

        if batch is None:

            # Treat input as one graph.
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device,
            )

        if batch.dim() != 1:
            raise ValueError(
                "batch must have shape [num_nodes]"
            )

        if batch.size(0) != x.size(0):
            raise ValueError(
                "batch must contain one value per node"
            )

        # =========================================================
        # GAT message passing
        # =========================================================

        h = x

        for layer_index, conv in enumerate(self.convs):

            h = conv(
                h,
                edge_index,
                edge_attr,
            )

            # All layers except the final layer receive:
            # normalization -> activation -> dropout
            if layer_index < len(self.convs) - 1:

                h = self.norms[layer_index](h)

                h = F.elu(h)

                h = F.dropout(
                    h,
                    p=self.dropout,
                    training=self.training,
                )

        # =========================================================
        # Final node representation
        # =========================================================

        h = self.output_norm(h)

        # =========================================================
        # Global mean pooling
        # =========================================================
        #
        # Variable number of nodes:
        #
        # Graph A -> 11 nodes
        # Graph B -> 22 nodes
        # Graph C -> 50 nodes
        #
        # All become:
        #
        # [1, graph_embedding_dim]
        #
        # when processed individually.
        # =========================================================

        graph_embedding = global_mean_pool(
            h,
            batch,
        )

        return graph_embedding