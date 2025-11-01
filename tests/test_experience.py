import numpy as np
from src.experience.episodic import EpisodicMemory

def test_episodic_memory():
    """
    Tests the EpisodicMemory component.
    """
    memory = EpisodicMemory(capacity=10)

    # Remember some experiences
    for i in range(15):
        memory.remember(
            state=np.random.rand(10),
            action=np.random.randint(0, 3),
            outcome=np.random.rand(),
            surprise=np.random.rand()
        )

    # Check that the memory has the correct number of episodes
    assert len(memory.episodes) == 10

    # Recall some experiences
    recalled_episodes = memory.recall(current_state=np.random.rand(10), k=5)

    # Check that the correct number of episodes were recalled
    assert len(recalled_episodes) == 5
