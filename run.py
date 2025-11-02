import yaml
import torch
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any
from sklearn.preprocessing import MinMaxScaler

from src.intelligence.multitask import MultiTaskPredictor
from src.data.unified_pipeline import DataIntelligence
from src.experience.episodic import EpisodicMemory
from src.improvement.analyzer import SelfImprovement
from src.core.nas import EfficientNAS

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("live_test.log"),
                              logging.StreamHandler()])

class EliteSystem:
    """
    The main orchestrator for the Elite Adaptive Prediction System.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initializes the EliteSystem.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logging.info("Configuration loaded.")

        # Initialize components
        self.brain = self._build_brain()
        self.data = DataIntelligence()
        self.experience = EpisodicMemory(capacity=self.config['experience']['capacity'])
        self.improvement = SelfImprovement(model=self.brain)
        self.nas = EfficientNAS(input_dim=5, output_dim=1)
        self.scaler = MinMaxScaler()
        logging.info("System components initialized.")

        # Performance metrics
        self.correct_predictions = 0
        self.total_predictions = 0

    def _build_brain(self) -> Any:
        """
        Builds the core brain of the system.
        """
        self.input_dim = 5  # O, H, L, C, Vol
        model = MultiTaskPredictor(
            input_dim=self.input_dim,
            hidden_dim=self.config['brain']['encoder_dim'],
            dropout_rate=self.config['brain']['dropout_rate']
        )
        return model

    @staticmethod
    def _preprocess_data(data: pd.DataFrame, scaler: MinMaxScaler, seq_len: int = 10) -> torch.Tensor:
        """Preprocesses DataFrame into a tensor for the model."""
        features = data[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
        features = scaler.transform(features)

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
        logging.info("Starting adaptive learning loop.")

        # Fit the scaler on initial data
        initial_data = self.data.collect(asset='AAPL', lookback='1y')
        self.scaler.fit(initial_data[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy())

        while True:
            start_time = time.time()
            try:
                # 1. Collect new data
                data = self.data.collect(asset='AAPL')

                if len(data) < 10:
                    logging.warning("Not enough data to proceed.")
                    time.sleep(60)
                    continue

                # Get the last closing price for accuracy calculation
                last_close = data['Close'].iloc[-1]
                if hasattr(last_close, 'item'):
                    last_close = last_close.item()

                # Preprocess data
                input_tensor = self._preprocess_data(data, self.scaler, seq_len=10)

                if input_tensor.nelement() == 0:
                    logging.warning("Could not create sequences from data.")
                    time.sleep(60)
                    continue

                # 2. Make prediction
                self.brain.eval()
                with torch.no_grad():
                    predictions = self.brain(input_tensor[-1].unsqueeze(0))

                price_pred_normalized = predictions['price'].item()

                # Create a dummy array to inverse transform the prediction
                dummy_array = np.zeros((1, 5))
                dummy_array[0, 3] = price_pred_normalized  # Close price is the 4th feature
                price_pred = self.scaler.inverse_transform(dummy_array)[0, 3]

                logging.info(f"Last close: {last_close:.2f}, Predicted next close: {price_pred:.2f}")

                # Wait for a minute to get the actual next price
                time.sleep(60)

                # 4. Get actual outcome
                actual_close = self.data.get_live_price(asset='AAPL')
                if actual_close == 0.0:
                    logging.warning("Could not retrieve live price. Skipping accuracy calculation.")
                    continue

                logging.info(f"Actual close: {actual_close:.2f}")

                # 5. Calculate directional accuracy
                if price_pred > last_close:
                    predicted_direction = 1
                elif price_pred < last_close:
                    predicted_direction = -1
                else:
                    predicted_direction = 0

                if actual_close > last_close:
                    actual_direction = 1
                elif actual_close < last_close:
                    actual_direction = -1
                else:
                    actual_direction = 0

                if predicted_direction == actual_direction:
                    self.correct_predictions += 1
                self.total_predictions += 1

                accuracy = (self.correct_predictions / self.total_predictions) * 100
                logging.info(f"Directional Accuracy: {accuracy:.2f}% ({self.correct_predictions}/{self.total_predictions})")

                # Self-improvement
                self.experience.add( (input_tensor[-1], torch.FloatTensor([actual_close])) )
                if len(self.experience.memory) > 50:
                    recent_data = self.experience.sample(32)
                    self.improvement.apply_improvements(recent_data)
                    logging.info("Applied self-improvement.")

            except Exception as e:
                logging.error(f"An error occurred: {e}", exc_info=True)

            latency = time.time() - start_time
            logging.info(f"Loop latency: {latency:.2f} seconds.")

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