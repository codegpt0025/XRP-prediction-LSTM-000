import torch
import pytest
from src.core.encoder import AdaptiveEncoder, AttentionPooling

@pytest.fixture
def sample_data():
    """Provides a sample tensor for testing."""
    return torch.randn(2, 10, 50)  # batch_size=2, seq_len=10, input_dim=50

def test_attention_pooling_shape(sample_data):
    """Tests that the AttentionPooling layer produces the correct output shape."""
    pooling = AttentionPooling(input_dim=50)
    pooled = pooling(sample_data)
    assert pooled.shape == (2, 50), f"Expected shape (2, 50), but got {pooled.shape}"

def test_adaptive_encoder_forward_pass(sample_data):
    """Tests that the AdaptiveEncoder can perform a forward pass without errors."""
    encoder = AdaptiveEncoder(input_dim=50, hidden_dim=64, num_heads=4, num_layers=2)
    try:
        output = encoder(sample_data)
    except Exception as e:
        pytest.fail(f"AdaptiveEncoder forward pass failed with an exception: {e}")
    assert output.shape == (2, 64), f"Expected shape (2, 64), but got {output.shape}"
