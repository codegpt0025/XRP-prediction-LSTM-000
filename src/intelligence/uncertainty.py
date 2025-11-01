import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class UncertaintyEstimator:
    """
    Estimates the uncertainty of a model's predictions using techniques like
    Monte Carlo (MC) Dropout.
    """

    def __init__(self, model: nn.Module, n_forward_passes: int = 20):
        """
        Initializes the UncertaintyEstimator.

        Args:
            model (nn.Module): The model for which to estimate uncertainty.
            n_forward_passes (int, optional): The number of forward passes for MC Dropout.
                                              Defaults to 20.
        """
        self.model = model
        self.n_forward_passes = n_forward_passes

    def estimate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimates the mean prediction, epistemic uncertainty, and aleatoric uncertainty.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the mean prediction,
                                                            epistemic uncertainty, and aleatoric uncertainty.
        """
        # Enable dropout during inference
        self.model.train()

        predictions = []
        for _ in range(self.n_forward_passes):
            with torch.no_grad():
                predictions.append(self.model(x))

        predictions = torch.stack(predictions)

        mean_prediction = predictions.mean(dim=0)

        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = predictions.var(dim=0)

        # Aleatoric uncertainty (data uncertainty) - requires the model to output variance
        # This is a simplified placeholder.
        aleatoric_uncertainty = torch.zeros_like(mean_prediction)

        return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty
