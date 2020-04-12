import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int, fc1_units: int = 64, fc2_units: int = 64) -> None:
        """Initialize parameters and build model.

        :param state_size: Dimension of state space
        :param action_size: Dimension of action space
        :param seed: Random seed
        :param fc1_units: Dimension of first hidden layer output
        :param fc2_units: Dimension of second hidden layer output
        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map state to action values."""

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
