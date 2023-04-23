import gymnasium as gym
import numpy as np


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

    def act(self, state: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Returns actions for given state as per current policy.
        Args:

        state (list): The state. Attributes:
            state[0] is the horizontal coordinate
            state[1] is the vertical coordinate
            state[2] is the horizontal speed
            state[3] is the vertical speed
            state[4] is the angle
            state[5] is the angular speed
            state[6] 1 if first leg has contact, else 0
            state[7] 1 if second leg has contact, else 0

        from: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
        """

        angle_targ = (
            state[0] * 0.5 + state[2] * 1.0
        )  # angle should point towards center

        threshold = 0.41

        if angle_targ > threshold:
            angle_targ = threshold  # more than threshold is bad
        if angle_targ < -threshold:
            angle_targ = -threshold
        hover_targ = 0.55 * np.abs(
            state[0]
        )  # target y should be proportional to horizontal offset

        angle_todo = (angle_targ - state[4]) * 0.5 - (state[5]) * 1.0
        hover_todo = (hover_targ - state[1]) * 0.5 - (state[3]) * 0.5

        if state[6] or state[7]:  # legs have contact
            angle_todo = 0
            hover_todo = (
                -(state[3]) * 0.5
            )  # override to reduce fall speed, that's all we need after contact

        action = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            action = 2
        elif angle_todo < -0.05:
            action = 3
        elif angle_todo > +0.05:
            action = 1

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
        """
        pass
