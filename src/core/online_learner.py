import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List

class OnlineLearner:
    """
    Implements an online learning mechanism using Elastic Weight Consolidation (EWC)
    to prevent catastrophic forgetting. The learner identifies and protects important
    parameters learned from past data.
    """

    def __init__(self, model: nn.Module, alpha: float = 0.5):
        """
        Initializes the OnlineLearner.

        Args:
            model (nn.Module): The model to be trained online.
            alpha (float, optional): The weighting factor for the EWC loss. Defaults to 0.5.
        """
        self.model = model
        self.alpha = alpha
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def _compute_fisher_information(self, data: List[torch.Tensor]):
        """
        Computes the Fisher Information Matrix, which estimates the importance of each parameter.

        Args:
            data (List[torch.Tensor]): A list of data tensors for Fisher computation.
        """
        # Create a DataLoader for the provided data
        dataset = TensorDataset(*data)
        dataloader = DataLoader(dataset, batch_size=32)

        # Initialize Fisher Information Matrix
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        self.model.train()
        for batch in dataloader:
            inputs, targets = batch
            self.model.zero_grad()
            outputs = self.model(inputs)
            # Use a simplified loss for Fisher computation
            loss = nn.MSELoss()(outputs['price'], targets)
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        # Average the Fisher Information
        for name in fisher:
            fisher[name] /= len(dataloader)

        self.fisher_information = fisher

    def consolidate(self, data: List[torch.Tensor]):
        """
        Consolidates knowledge by computing and storing the Fisher Information Matrix
        and the optimal parameters from the current data.

        Args:
            data (List[torch.Tensor]): The data to consolidate knowledge from.
        """
        self._compute_fisher_information(data)
        self.optimal_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def ewc_loss(self) -> torch.Tensor:
        """
        Calculates the EWC loss, which penalizes changes to important parameters.
        """
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal_param = self.optimal_params[name]
                loss += (fisher * (param - optimal_param).pow(2)).sum()
        return self.alpha * loss
