import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


class DQN(nn.Module):
    """
    DQN Model.
    """

    def __init__(self, state_size, action_size):
        """
        Initialize a Deep Q-Network (DQN) model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """

        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        """
        Forward pass of the DQN model.
        """
        return self.fc(x)


class Agent:
    """
    My Agent Architecture
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Initialize RL agent.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.buffer = deque(maxlen=10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(observation_space.shape[0], action_space.n).to(self.device)
        self.target_model = DQN(observation_space.shape[0], action_space.n).to(
            self.device
        )
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_freq = 500
        self.steps = 0
        self.prev_state = None
        self.prev_action = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss_fn = nn.MSELoss()

    def act(self, state: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Returns actions for given state as per current policy.
        """

        current_state = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:

            with torch.no_grad():
                action = self.model(current_state).argmax().item()

        self.prev_state = state
        self.prev_action = action
        return action

    def learn(
        self,
        state: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Takes an observation, a reward, a boolean indicating whether the episode has terminated,
        and a boolean indicating whether the episode was truncated.

        Update value parameters using given batch of experience tuples.
        Q_targets = r + γ * Q_targets_next
        where:
            Q_targets_next = argmax_a(Q_target(next_state, a))
        """

        if self.prev_state is not None:
            self.buffer.append(
                (
                    self.prev_state,
                    self.prev_action,
                    reward,
                    state,
                    terminated,
                )
            )

        if len(self.buffer) < self.batch_size:
            return

        if self.steps % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps += 1

        batch = random.sample(self.buffer, self.batch_size)

        # Double DQN
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_model(next_states).gather(1, next_actions)
        target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if terminated or truncated:
            self.target_model.load_state_dict(self.model.state_dict())
