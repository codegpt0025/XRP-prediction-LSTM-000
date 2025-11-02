import torch
from torch import nn
from typing import Dict, Any

class AttentionPooling(nn.Module):
    """
    An attention-based pooling layer that learns a query vector to produce a weighted
    summary of a sequence. This allows the model to focus on the most relevant
    time steps.
    """
    def __init__(self, input_dim: int):
        """
        Initializes the AttentionPooling layer.

        Args:
            input_dim (int): The dimensionality of the input sequence.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AttentionPooling layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The pooled output of shape (batch_size, input_dim).
        """
        # Calculate attention scores
        attn_scores = torch.matmul(x, self.query.t()).squeeze(-1)
        # Convert scores to probabilities
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)
        # Calculate the weighted sum
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled

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

        # Configure the Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )

        # Stack the encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Add the attention pooling layer
        self.pooling = AttentionPooling(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the AdaptiveEncoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: The learned representation of the input data.
        """
        # 1. Project input to the hidden dimension
        x = self.input_layer(x)

        # 2. Pass through the Transformer encoder
        representation = self.transformer_encoder(x)

        # 3. Apply adaptive pooling to get the final representation
        pooled_representation = self.pooling(representation)

        return pooled_representation
