import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class MemoryAugmentedNetwork(nn.Module):
    """
    A neural network augmented with an external memory module, enabling it to store
    and retrieve information over long time scales.
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

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Reads from memory using content-based addressing.

        Args:
            query (torch.Tensor): The query vector to retrieve similar items from memory.

        Returns:
            torch.Tensor: The retrieved memory content.
        """
        # Compute cosine similarity between query and memory slots
        similarities = F.cosine_similarity(query.unsqueeze(1), self.memory.unsqueeze(0), dim=-1)

        # Get attention weights over memory
        attention_weights = F.softmax(similarities, dim=-1)

        # Retrieve content from memory
        retrieved_memory = torch.matmul(attention_weights.unsqueeze(1), self.memory).squeeze(1)

        return retrieved_memory

    def write(self, value: torch.Tensor, gate: torch.Tensor):
        """
        Writes to memory, updating existing patterns.

        Args:
            value (torch.Tensor): The new information to be written to memory.
            gate (torch.Tensor): A gating mechanism to control the write operation.
        """
        # For simplicity, this example uses a simple update rule.
        # A more advanced implementation might use separate erase and add gates.
        self.memory.data = self.memory.data * (1 - gate.unsqueeze(-1)) + value.unsqueeze(1) * gate.unsqueeze(-1)
