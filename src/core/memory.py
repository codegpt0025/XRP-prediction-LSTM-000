import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class MemoryAugmentedNetwork(nn.Module):
    """
    A neural network augmented with an external memory module, inspired by Differentiable
    Neural Computers. It can store and retrieve information over long time scales using
    a sophisticated write mechanism with explicit erase and add gates.
    """

    def __init__(self, memory_slots: int = 256, slot_dim: int = 512):
        """
        Initializes the MemoryAugmentedNetwork.

        Args:
            memory_slots (int, optional): The number of memory slots. Defaults to 256.
            slot_dim (int, optional): The dimensionality of each memory slot. Defaults to 512.
        """
        super().__init__()
        self.memory_slots = memory_slots
        self.slot_dim = slot_dim

        # Initialize memory as a learnable parameter
        self.memory = nn.Parameter(torch.randn(memory_slots, slot_dim))

    def _get_addressing_weights(self, key: torch.Tensor) -> torch.Tensor:
        """
        Computes addressing weights based on cosine similarity.

        Args:
            key (torch.Tensor): The key vector of shape (batch_size, slot_dim).

        Returns:
            torch.Tensor: The attention weights over memory of shape (batch_size, memory_slots).
        """
        # Normalize memory and key for stable similarity scores
        normalized_memory = F.normalize(self.memory, p=2, dim=-1)
        normalized_key = F.normalize(key, p=2, dim=-1)

        # Compute cosine similarity
        similarities = torch.matmul(normalized_key, normalized_memory.t())

        # Get attention weights over memory
        attention_weights = F.softmax(similarities, dim=-1)
        return attention_weights

    def read(self, read_keys: torch.Tensor) -> torch.Tensor:
        """
        Reads from memory using content-based addressing.

        Args:
            read_keys (torch.Tensor): The query vectors of shape (batch_size, slot_dim).

        Returns:
            torch.Tensor: The retrieved memory content of shape (batch_size, slot_dim).
        """
        # Get read weights and retrieve memory
        read_weights = self._get_addressing_weights(read_keys)
        retrieved_memory = torch.matmul(read_weights, self.memory)
        return retrieved_memory

    def write(self, write_key: torch.Tensor, erase_vector: torch.Tensor, add_vector: torch.Tensor):
        """
        Writes to memory using a gated mechanism with separate erase and add operations.

        Args:
            write_key (torch.Tensor): The key to determine write locations, shape (batch_size, slot_dim).
            erase_vector (torch.Tensor): The vector to erase from memory, shape (batch_size, slot_dim).
            add_vector (torch.Tensor): The vector to add to memory, shape (batch_size, slot_dim).
        """
        # Get write weights
        write_weights = self._get_addressing_weights(write_key).unsqueeze(2) # (batch_size, memory_slots, 1)

        # Reshape erase and add vectors for broadcasting
        erase_vector = erase_vector.unsqueeze(1) # (batch_size, 1, slot_dim)
        add_vector = add_vector.unsqueeze(1) # (batch_size, 1, slot_dim)

        # Gated erase operation
        erase_mask = write_weights * torch.sigmoid(erase_vector)
        self.memory.data = self.memory.data * (1 - erase_mask.sum(0))

        # Gated add operation
        add_masked = write_weights * add_vector
        self.memory.data = self.memory.data + add_masked.sum(0)
