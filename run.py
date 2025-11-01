import yaml
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any

from src.intelligence.multitask import MultiTaskPredictor
from src.data.unified_pipeline import DataIntelligence
from src.experience.episodic import EpisodicMemory
from src.improvement.analyzer import SelfImprovement
from src.core.nas import EfficientNAS

class EliteSystem:
    """
    The main orchestrator for the Elite Adaptive Prediction System.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initializes the EliteSystem.

        Args:
            config_path (str, optional): The path to the configuration file.
                                         Defaults to 'config.yaml'.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.brain = self._build_brain()
        self.data = DataIntelligence()
        self.experience = EpisodicMemory(capacity=self.config['experience']['capacity'])
        self.improvement = SelfImprovement(model=self.brain)
        self.nas = EfficientNAS(input_dim=5, output_dim=1)

    def _build_brain(self) -> Any:
        """
        Builds the core brain of the system.
        """
        self.input_dim = 5  # O, H, L, C, Vol
        model = MultiTaskPredictor(
            input_dim=self.input_dim,
            hidden_dim=self.config['brain']['encoder_dim']
        )
        return model

    def _preprocess_data(self, data: pd.DataFrame, seq_len: int = 10) -> torch.Tensor:
        """Preprocesses DataFrame into a tensor for the model."""
        features = data[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()

        # Simple normalization
        if features.std(axis=0).all() != 0:
            features = (features - features.mean(axis=0)) / features.std(axis=0)

        sequences = []
        for i in range(len(features) - seq_len + 1):
            sequences.append(features[i:i + seq_len])

        if not sequences:
            return torch.empty(0)

        return torch.FloatTensor(np.array(sequences))

    def run_adaptive(self):
        """
        Main adaptive learning loop.
        """
        while True:
            # 1. Collect new data
            data = self.data.collect(asset='AAPL', lookback='1mo')

            if len(data) < 10:
                print("Not enough data to proceed.")
                break

            # Preprocess data
            input_tensor = self._preprocess_data(data, seq_len=10)

            if input_tensor.nelement() == 0:
                print("Could not create sequences from data.")
                break

            # 2. Make prediction
            self.brain.eval()
            with torch.no_grad():
                predictions = self.brain(input_tensor[-1].unsqueeze(0))

            price_pred = predictions['price'].item()
            print(f"Predicted Price: {price_pred}")

            # Other components are not yet integrated
            # 3. RL decision
            # 4. Wait for actual outcome
            # 5. Remember experience
            # 6. Online update
            # 7. Self-improve

            # 8. Display status
            self._display_status()

            # For demonstration, we'll break after one loop.
            break

    def run_nas(self, budget: int):
        """
        Neural architecture search.
        """
        best_arch = self.nas.search(validation_data=None, budget=budget)
        print(f"Best architecture found: {best_arch}")
        # self.brain.update_architecture(best_arch)

    def _display_status(self):
        """
        Displays the current status of the system.
        """
        print("System status: OK")

if __name__ == '__main__':
    system = EliteSystem()
    if system.config['system']['mode'] == 'adaptive':
        system.run_adaptive()
    elif system.config['system']['mode'] == 'nas':
        system.run_nas(budget=system.config['nas']['budget'])
