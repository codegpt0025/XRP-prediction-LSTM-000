import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import gym
from typing import Tuple

class RLDecisionAgent:
    """
    An agent that uses Reinforcement Learning (RL) to make optimal decisions
    based on the current market state.
    """

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, algorithm: str = 'PPO'):
        """
        Initializes the RLDecisionAgent.

        Args:
            observation_space (gym.spaces.Space): The observation space for the RL environment.
            action_space (gym.spaces.Space): The action space for the RL environment.
            algorithm (str, optional): The RL algorithm to use. Defaults to 'PPO'.
        """
        if algorithm.upper() == 'PPO':
            self.model = PPO(ActorCriticPolicy, env=None, observation_space=observation_space, action_space=action_space)
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not supported.")

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Selects an action based on the current state.

        Args:
            state (torch.Tensor): The current market state.

        Returns:
            Tuple[int, torch.Tensor]: A tuple containing the selected action and its confidence.
        """
        action, _ = self.model.predict(state, deterministic=True)

        # Confidence can be derived from the action probabilities
        action_probabilities = self.model.policy.predict_values(state)
        confidence = torch.max(action_probabilities)

        return action, confidence

    def train(self, env: gym.Env, total_timesteps: int = 10000):
        """
        Trains the RL agent.

        Args:
            env (gym.Env): The training environment.
            total_timesteps (int, optional): The total number of timesteps to train for. Defaults to 10000.
        """
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps)
