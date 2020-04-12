from collections import namedtuple, deque
import torch
import random
import numpy as np
from typing import List, Tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int) -> None:
        """Initialize a ReplayBuffer object.

        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param seed: random seed
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def __len__(self) -> int:
        """Return the current size of internal memory."""

        return len(self.memory)

    def add(self, state: np.ndarray, action: np.ndarray, reward: List[float], next_state: np.ndarray, done: bool) -> None:
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    # noinspection PyTypeChecker
    def sample(self) -> Tuple[torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

