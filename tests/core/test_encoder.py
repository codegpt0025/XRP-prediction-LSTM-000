import torch
from src.core.encoder import AdaptiveEncoder

def test_adaptive_encoder():
    """
    Tests the AdaptiveEncoder.
    """
    input_dim = 10
    hidden_dim = 64
    batch_size = 4
    seq_len = 8

    encoder = AdaptiveEncoder(input_dim, hidden_dim)

    # Create a dummy input tensor
    x = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    output = encoder(x)

    # Check output shape
    assert output.shape == (batch_size, hidden_dim)
