import numpy as np
from src.experience.episodic import EpisodicMemory

def test_episodic_memory():
    """
    Tests the EpisodicMemory component.
    """
    memory = EpisodicMemory(capacity=10)

    # Add some experiences
    for i in range(15):
        memory.add(
            (np.random.rand(10), np.random.rand())
        )

    # Check that the memory has the correct number of episodes
    assert len(memory.memory) == 10

    # Sample some experiences
    sampled_episodes = memory.sample(5)

    # Check that the correct number of episodes were recalled
    assert len(sampled_episodes) == 5
