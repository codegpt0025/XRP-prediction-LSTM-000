import gym
import numpy as np
import torch
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Tuple, Dict, Any

class MarketEnv(gym.Env):
    """
    A custom gym environment for simulating market dynamics for the RLDecisionAgent.
    The state includes market features, uncertainty, and memory recall.
    """
    def __init__(self, state_dim: int, initial_balance: float = 10000.0):
        super(MarketEnv, self).__init__()
        self.state_dim = state_dim
        self.initial_balance = initial_balance
        self.balance = initial_balance

        # Actions: {predict_high, predict_low, wait, uncertain}
        self.action_space = spaces.Discrete(4)

        # Observation space: market features, uncertainty, memory recall
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.
        """
        # Simulate market movement (placeholder for real data integration)
        market_movement = np.random.choice([-0.01, 0.01, 0.0]) # Down, Up, Flat

        # Reward function based on Sharpe ratio and directional accuracy
        reward = self._calculate_reward(action, market_movement)
        self.balance += reward

        # Generate the next state (placeholder)
        next_state = np.random.randn(self.state_dim).astype(np.float32)

        # Check if the episode is done (e.g., balance drops to zero)
        done = self.balance <= 0

        return next_state, reward, done, {}

    def _calculate_reward(self, action: int, market_movement: float) -> float:
        """
        Calculates the reward for a given action and market movement.
        """
        if (action == 0 and market_movement > 0) or \
           (action == 1 and market_movement < 0):
            return 1.0  # Correct prediction
        elif (action == 0 and market_movement < 0) or \
             (action == 1 and market_movement > 0):
            return -1.0 # Incorrect prediction
        elif action == 2:
            return 0.1 # Small reward for waiting
        return 0.0 # No reward/penalty for being uncertain

    def reset(self) -> np.ndarray:
        """
        Resets the state of the environment to an initial state.
        """
        self.balance = self.initial_balance
        # Return initial state
        return np.random.randn(self.state_dim).astype(np.float32)

class RLDecisionAgent:
    """
    An agent that uses Reinforcement Learning (RL) to make optimal decisions
    based on the current market state.
    """
    def __init__(self, env: gym.Env, algorithm: str = 'PPO'):
        """
        Initializes the RLDecisionAgent.

        Args:
            env (gym.Env): The gym environment.
            algorithm (str, optional): The RL algorithm to use. Defaults to 'PPO'.
        """
        if algorithm.upper() == 'PPO':
            self.model = PPO(ActorCriticPolicy, env)
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not supported.")

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Selects an action based on the current state.

        Args:
            state (np.ndarray): The current market state.

        Returns:
            Tuple[int, torch.Tensor]: A tuple containing the selected action and its confidence.
        """
        action, _ = self.model.predict(state, deterministic=True)

        # Get action probabilities for confidence
        state_tensor = torch.as_tensor(state).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(state_tensor)
            probs = dist.distribution.probs
            confidence = torch.max(probs)

        return int(action), confidence

    def train(self, total_timesteps: int = 10000):
        """
        Trains the RL agent.

        Args:
            total_timesteps (int, optional): The total number of timesteps to train for.
                                            Defaults to 10000.
        """
        self.model.learn(total_timesteps=total_timesteps)
