from typing import List
import numpy as np

class AdaptiveTrainer:
    """
    Trains the model when needed, based on performance degradation, rather than
    on a fixed schedule.
    """

    def __init__(self, model, training_threshold: float = 0.5):
        """
        Initializes the AdaptiveTrainer.

        Args:
            model: The predictive model.
            training_threshold (float, optional): The performance threshold below which
                                                   training is triggered. Defaults to 0.5.
        """
        self.model = model
        self.training_threshold = training_threshold

    def should_train(self, performance_history: List[float]) -> bool:
        """
        Detects if the model's performance has degraded.

        Args:
            performance_history (List[float]): A history of the model's performance.

        Returns:
            bool: True if training is needed, False otherwise.
        """
        if len(performance_history) < 2:
            return False

        # Trigger training if performance drops below the threshold
        return performance_history[-1] < self.training_threshold

    def train_efficiently(self, data):
        """
        Trains the model using sample-efficient techniques.

        Args:
            data: The training data.
        """
        # This would involve techniques like importance sampling (focusing on hard examples),
        # few-shot learning, and transfer learning.

        # For now, this is a placeholder for a more advanced training loop.
        self.model.train(data)
