import numpy as np
from hmmlearn import hmm
from typing import Tuple

class RegimeDetector:
    """
    Detects the current market regime using a Hidden Markov Model (HMM).
    """

    def __init__(self, n_regimes: int = 5):
        """
        Initializes the RegimeDetector.

        Args:
            n_regimes (int, optional): The number of market regimes to model. Defaults to 5.
        """
        self.n_regimes = n_regimes
        self.hmm = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100)

    def fit(self, market_data: np.ndarray):
        """
        Fits the HMM to the market data.

        Args:
            market_data (np.ndarray): A time series of market data.
        """
        self.hmm.fit(market_data)

    def detect(self, market_data: np.ndarray) -> Tuple[int, float]:
        """
        Detects the current market regime.

        Args:
            market_data (np.ndarray): The most recent market data.

        Returns:
            Tuple[int, float]: A tuple containing the current regime and the confidence of the prediction.
        """
        hidden_states = self.hmm.predict(market_data)

        # The last state is the current regime
        current_regime = hidden_states[-1]

        # Confidence can be estimated from the posterior probabilities
        posterior_probs = self.hmm.predict_proba(market_data)
        confidence = posterior_probs[-1, current_regime]

        return current_regime, confidence

    def get_regime_strategy(self, regime: int) -> str:
        """
        Returns the appropriate strategy for a given regime.
        """
        # This would be learned, not hardcoded.
        regime_map = {
            0: "trending",
            1: "mean-reverting",
            2: "volatile",
            3: "quiet",
            4: "crisis"
        }
        return regime_map.get(regime, "unknown")
