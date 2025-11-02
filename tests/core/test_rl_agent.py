import gym
import numpy as np
import pytest
import torch
from src.core.rl_agent import MarketEnv, RLDecisionAgent

@pytest.fixture
def market_env():
    """Provides a MarketEnv instance for testing."""
    return MarketEnv(state_dim=10)

@pytest.fixture
def rl_agent(market_env):
    """Provides an RLDecisionAgent instance for testing."""
    return RLDecisionAgent(env=market_env)

def test_action_selection(rl_agent, market_env):
    """Tests that the RLDecisionAgent can select an action."""
    state = market_env.reset()
    try:
        action, confidence = rl_agent.select_action(state)
    except Exception as e:
        pytest.fail(f"RLDecisionAgent action selection failed with an exception: {e}")
    assert isinstance(action, int)
    assert action in market_env.action_space
    assert isinstance(confidence, torch.Tensor)

def test_training_loop(rl_agent):
    """Tests that the RLDecisionAgent training loop runs without errors."""
    try:
        rl_agent.train(total_timesteps=10)
    except Exception as e:
        pytest.fail(f"RLDecisionAgent training loop failed with an exception: {e}")
