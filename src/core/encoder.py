import torch
from torch import nn
from typing import Dict, Any

class AdaptiveEncoder(nn.Module):
    """
    Learns the optimal feature representation from raw inputs using a Transformer-based architecture.
    This encoder is designed for automatic feature discovery, eliminating the need for manual feature engineering.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_heads: int = 8, num_layers: int = 6):
        """
        Initializes the AdaptiveEncoder.

        Args:
            input_dim (int): The dimensionality of the input data.
            hidden_dim (int, optional): The dimensionality of the hidden layers. Defaults to 512.
            num_heads (int, optional): The number of attention heads in the Transformer encoder. Defaults to 8.
            num_layers (int, optional): The number of layers in the Transformer encoder. Defaults to 6.
        """
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AdaptiveEncoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The learned representation of the input data.
        """
        # Project input to hidden dimension
        x = self.input_layer(x)

        # Pass through Transformer encoder
        representation = self.transformer_encoder(x)

        # Adaptive pooling (placeholder for now, can be implemented with another attention layer)
        pooled_representation = representation.mean(dim=1)

        return pooled_representation
