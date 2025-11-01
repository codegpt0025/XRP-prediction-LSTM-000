import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import higher
from typing import List

class MAMLMetaLearner:
    """
    Implements Model-Agnostic Meta-Learning (MAML) to train a model that can
    quickly adapt to new tasks.
    """

    def __init__(self, model: nn.Module, inner_lr: float = 0.01, meta_lr: float = 0.001):
        """
        Initializes the MAMLMetaLearner.

        Args:
            model (nn.Module): The model to be meta-trained.
            inner_lr (float, optional): The learning rate for the inner loop. Defaults to 0.01.
            meta_lr (float, optional): The learning rate for the outer loop. Defaults to 0.001.
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)

    def meta_train(self, task_distribution: List[DataLoader], num_epochs: int):
        """
        Performs meta-training over a distribution of tasks.

        Args:
            task_distribution (List[DataLoader]): A list of DataLoaders, each representing a task.
            num_epochs (int): The number of meta-training epochs.
        """
        for epoch in range(num_epochs):
            meta_loss = 0.0
            for task in task_distribution:
                meta_loss += self.inner_loop(task)

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

    def inner_loop(self, task: DataLoader) -> torch.Tensor:
        """
        Performs the inner loop of MAML for a single task.

        Args:
            task (DataLoader): The DataLoader for the current task.

        Returns:
            torch.Tensor: The loss on the query set for the task.
        """
        with higher.innerloop_ctx(self.model, self.meta_optimizer, copy_initial_weights=True) as (fmodel, diffopt):
            # Adapt to the support set
            for x_support, y_support in task:
                y_pred = fmodel(x_support)
                support_loss = nn.MSELoss()(y_pred, y_support)
                diffopt.step(support_loss)

            # Evaluate on the query set
            x_query, y_query = next(iter(task))
            y_pred = fmodel(x_query)
            query_loss = nn.MSELoss()(y_pred, y_query)

        return query_loss
