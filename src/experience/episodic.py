import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any

class EpisodicMemory:
    """
    A human-like memory system that stores and retrieves experiences (episodes)
    based on their content and importance.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initializes the EpisodicMemory.

        Args:
            capacity (int, optional): The maximum number of episodes to store.
                                      Defaults to 100000.
        """
        self.capacity = capacity
        self.episodes: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None

    def _compute_importance(self, surprise: float, outcome: float) -> float:
        """
        Computes the importance of an episode based on surprise and outcome.
        """
        return abs(surprise * outcome)

    def remember(self, state: np.ndarray, action: Any, outcome: float, surprise: float):
        """
        Stores a new episode in memory.

        Args:
            state (np.ndarray): The state associated with the episode.
            action (Any): The action taken.
            outcome (float): The outcome of the action.
            surprise (float): The measure of surprise for the outcome.
        """
        if len(self.episodes) >= self.capacity:
            # Evict the least important episode
            self.episodes.pop(0)
            if self.embeddings is not None:
                self.embeddings = self.embeddings[1:]

        episode = {
            'state': state,
            'action': action,
            'outcome': outcome,
            'surprise': surprise,
            'importance': self._compute_importance(surprise, outcome)
        }
        self.episodes.append(episode)

        # Update embeddings
        state_embedding = state.reshape(1, -1)
        if self.embeddings is None:
            self.embeddings = state_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, state_embedding])

    def recall(self, current_state: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Recalls the k most similar past experiences.

        Args:
            current_state (np.ndarray): The current state to find similar experiences for.
            k (int, optional): The number of experiences to recall. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of the k most similar episodes.
        """
        if not self.episodes:
            return []

        similarities = cosine_similarity(current_state.reshape(1, -1), self.embeddings).flatten()
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [self.episodes[i] for i in top_k_indices]
