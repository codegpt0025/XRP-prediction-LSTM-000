import torch
import pytest
from src.core.memory import MemoryAugmentedNetwork

@pytest.fixture
def memory_network():
    """Provides a MemoryAugmentedNetwork instance for testing."""
    return MemoryAugmentedNetwork(memory_slots=10, slot_dim=20)

@pytest.fixture
def sample_tensors():
    """Provides sample tensors for testing."""
    batch_size = 2
    slot_dim = 20
    read_keys = torch.randn(batch_size, slot_dim)
    write_key = torch.randn(batch_size, slot_dim)
    erase_vector = torch.randn(batch_size, slot_dim)
    add_vector = torch.randn(batch_size, slot_dim)
    return read_keys, write_key, erase_vector, add_vector

def test_read_operation(memory_network, sample_tensors):
    """Tests the read operation of the MemoryAugmentedNetwork."""
    read_keys, _, _, _ = sample_tensors
    try:
        retrieved_memory = memory_network.read(read_keys)
    except Exception as e:
        pytest.fail(f"MemoryAugmentedNetwork read operation failed with an exception: {e}")
    assert retrieved_memory.shape == (2, 20), f"Expected shape (2, 20), but got {retrieved_memory.shape}"

def test_write_operation(memory_network, sample_tensors):
    """Tests the write operation of the MemoryAugmentedNetwork."""
    _, write_key, erase_vector, add_vector = sample_tensors
    initial_memory = memory_network.memory.data.clone()
    try:
        memory_network.write(write_key, erase_vector, add_vector)
    except Exception as e:
        pytest.fail(f"MemoryAugmentedNetwork write operation failed with an exception: {e}")
    assert not torch.equal(initial_memory, memory_network.memory.data), "Memory data should have been modified after write operation."
