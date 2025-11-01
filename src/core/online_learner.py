import torch
from torch import nn, optim
from typing import Dict

class OnlineLearner:
    """
    Handles continuous learning without catastrophic forgetting, using techniques
    like Elastic Weight Consolidation (EWC).
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, ewc_lambda: float = 0.1):
        """
        Initializes the OnlineLearner.

        Args:
            model (nn.Module): The model to be trained online.
            optimizer (optim.Optimizer): The optimizer for the model.
            ewc_lambda (float, optional): The hyperparameter for EWC. Defaults to 0.1.
        """
        self.model = model
        self.optimizer = optimizer
        self.ewc_lambda = ewc_lambda
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}

    def update(self, new_batch):
        """
        Performs an online update on a new batch of data.

        Args:
            new_batch: The new data batch.
        """
        self.model.train()
        x, y = new_batch
        y_pred = self.model(x)
        loss = nn.MSELoss()(y_pred, y)

        # Add EWC penalty
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                opt_param = self.optimal_params[name]
                loss += (fisher * (param - opt_param) ** 2).sum() * self.ewc_lambda

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def consolidate(self, dataset):
        """
        Computes the Fisher Information Matrix to identify important parameters.
        """
        self.model.eval()
        # Store current optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()

        # Compute Fisher Information
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param.data)

        for x, y in dataset:
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = nn.MSELoss()(y_pred, y)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data.clone() ** 2 / len(dataset)
