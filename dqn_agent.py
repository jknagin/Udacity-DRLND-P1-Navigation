import numpy as np
import random
from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from replay_buffer import ReplayBuffer

GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        """Initialize an Agent object.

        :param state_size: dimension of each state
        :param action_size: dimension of each action
        :param seed: random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state: np.ndarray, eps: float = 0.) -> np.ndarray:
        """Return actions for given state as per current policy.

        :param state: current state
        :param eps: epsilon-greedy parameter [0, 1)
        :return: action for given state as per current policy
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def hard_update(self) -> None:
        """Assign weights of local network to target network."""

        self.soft_update(tau=1.0)

    def learn(self, experiences: Tuple[torch.Tensor], gamma: float) -> None:
        """Update value parameters using given batch of experience tuples.

        :param experiences: tuple of (s, a, r, s', done) tuples
        :param gamma: discount factor
        """

        # noinspection PyTupleAssignmentBalance
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(TAU)

    def load(self, solution_filename: str) -> None:
        """Load weights from existing weights file into local network.
        :param solution_filename: weights filename to load from
        """

        self.qnetwork_local.load_state_dict(torch.load(solution_filename))
        self.hard_update()

    def save(self, solution_filename: str = 'solution.pth') -> None:
        """Save local network weights.
        :param solution_filename: weights filename to save to
        """

        torch.save(self.qnetwork_local.state_dict(), solution_filename)

    def soft_update(self, tau: float) -> None:
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param tau: soft update parameter (0, 1]
        """

        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def step(self, state: np.ndarray, action: np.ndarray, reward: List[float], next_state: np.ndarray,
             done: bool) -> None:
        """ Save the (S, A, R, S, done) tuple to the internal experience replay buffer. Update model weights.

        :param state: current state
        :param action: chosen action
        :param reward: reward for choosing action at state
        :param next_state: next state of environment after taking action at previous state
        :param done: indication of whether next state is terminal (True) or not (False)
        """

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
