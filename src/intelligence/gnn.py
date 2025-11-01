import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class MarketGNN(nn.Module):
    """
    A Graph Neural Network (GNN) that reasons about the structure of the market,
    where assets are nodes and their correlations are edges.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """
        Initializes the MarketGNN.

        Args:
            in_channels (int): The number of input features for each node.
            hidden_channels (int): The number of hidden units in the GCN layers.
            out_channels (int): The number of output features for each node.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the MarketGNN.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The graph connectivity in COO format.
            edge_weight (torch.Tensor, optional): The edge weights. Defaults to None.

        Returns:
            torch.Tensor: The relationally-aware embeddings for each node.
        """
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight)
        return x
