import torch
import torch.nn as nn
from src.core.encoder import AdaptiveEncoder
from typing import Dict

class MultiTaskPredictor(nn.Module):
    """
    A model that predicts multiple objectives simultaneously using a shared encoder.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, n_regimes: int = 5, dropout_rate: float = 0.2):
        """
        Initializes the MultiTaskPredictor.

        Args:
            input_dim (int): The dimensionality of the input data.
            hidden_dim (int, optional): The dimensionality of the hidden layers. Defaults to 512.
            n_regimes (int, optional): The number of market regimes. Defaults to 5.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.2.
        """
        super().__init__()

        self.shared_encoder = AdaptiveEncoder(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.price_head = nn.Linear(hidden_dim, 1)
        self.volatility_head = nn.Linear(hidden_dim, 1)
        self.regime_head = nn.Linear(hidden_dim, n_regimes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the MultiTaskPredictor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of predictions for each task.
        """
        features = self.shared_encoder(x)
        features = self.dropout(features)

        return {
            'price': self.price_head(features),
            'volatility': self.volatility_head(features),
            'regime': self.regime_head(features)
        }
