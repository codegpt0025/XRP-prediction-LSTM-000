import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any

import random

class EpisodicMemory:
    """
    A simple memory system that stores and retrieves experiences.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initializes the EpisodicMemory.
        """
        self.capacity = capacity
        self.memory = []

    def add(self, experience):
        """
        Adds an experience to the memory.
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(experience)

    def sample(self, batch_size: int):
        """
        Samples a batch of experiences from the memory.
        """
        return random.sample(self.memory, batch_size)
