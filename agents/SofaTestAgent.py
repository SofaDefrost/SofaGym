import random
from abc import ABC
from statistics import mean
from typing import Optional

import gym
import numpy as np
import torch as th
from agents.utils import mkdirp


class SofaTestAgent(ABC):
    """
    Test class for creating and testing reinforcement learning policies in a SOFA scene 
    with gym environment.

    Parameters
    ----------
    env_id : str
        The name of the environment to be tested.
    seed : int, default=0
        The seed used for random number generation to initialize the environment.
    output_dir : str, default="./Results"
        The output directory to save the environment results.
    max_episode_steps : int, default=None
        The maximum number of steps that an agent can take in an episode of the environment.
        Once this limit is reached, the episode is terminated. If None, the episode will 
        continue until the goal is done or a termination conidition happens.
    
    Attributes
    ----------
    env_id : str
        The name of the environment to be tested.
    seed : int
        The seed used for random number generation to initialize the environment.
    output_dir : str
        The output directory to save the environment results.
    max_episode_steps : int
        The maximum number of steps that an agent can take in an episode of the environment.
        Once this limit is reached, the episode is terminated. If None, the episode will 
        continue until the goal is done or a termination conidition happens.
    test_env: gym.Env
        The test environment instance.
    
    Notes:
    -----
        This class is used to only test the environment and the agent without any 
        reinforcement learning training done.    

    Usage:
    -----
        Use the reset method before launching the environment.
    """
    def __init__(
        self,
        env_id: str,
        seed: Optional[int] = 0,
        output_dir: Optional[str] = "./Results",
        max_episode_steps: Optional[int] = None,
    ):
        """
        Initialization of test agent class. Creates a test environment for the SOFA scene.

        Parameters
        ----------
        env_id : str
            The name of the environment to be tested.
        seed : int, default=0
            The seed used for random number generation to initialize the environment.
        output_dir : str, default="./Results"
            The output directory to save the environment results.
        max_episode_steps : int, default=None
            The maximum number of steps that an agent can take in an episode of the environment.
            Once this limit is reached, the episode is terminated. If None, the episode will 
            continue until the goal is done or a termination conidition happens.
        """
        self.env_id = env_id
        self.seed = seed
        self.output_dir = output_dir
        self.max_episode_steps = max_episode_steps

        self.test_env = self.env_make()

    @property
    def seed(self):
        """
        Getter method for the seed used for the environment.
        
        Returns
        -------
        _seed: int
            The current seed used.
        """
        return self._seed

    @seed.setter
    def seed(self, seed):
        """
        Setter method to set the seed used for random number generators in Python, PyTorch, 
        and NumPy.
        
        Parameters
        ----------
        seed: int
            The seed value to set for the random number generators.
        """
        self._seed = seed
        random.seed(self._seed)
        th.manual_seed(self._seed)
        np.random.seed(self._seed)

    @property
    def output_dir(self):
        """
        Getter method for the output directory.
        
        Returns
        -------
        _output_dir: str
            The output directory to save results.
        """
        return self._output_dir

    @output_dir.setter
    def output_dir(self, output_dir):
        """
        Setter method to set the output directory for saving results and create the directory 
        if it doesn't exist.
        
        Parameters
        ----------
        output_dir: str
            The path of the output directory where results will be saved.
        """
        self._output_dir = output_dir
        mkdirp(self._output_dir)

    def env_make(self):
        """
        Create a gym environment for the provided scene initialized with the set `seed` and 
        wrapped using TimeLimit to limit the maximum number of steps per episode that 
        an agent can take according to `max_episode_steps`.
        
        Returns
        -------
        env: gym.Env
            the created instance of the gym environment.
        """
        env = gym.make(self.env_id)

        if self.max_episode_steps:
            env = gym.wrappers.TimeLimit(env, self.max_episode_steps)

        self.env_seed(env)

        return env

    def env_seed(self, env):
        """
        Set the seed for the environment and its action space.
        
        Parameters
        ----------
        env: gym.Env
            The gym environment instance to set the seed for.
        
        """
        env.seed(self.seed)
        env.action_space.seed(self.seed)
        env.action_space.np_random.seed(self.seed)

    def policy(self, obs):
        """
        Choose an action for the agent to take in the environment according to 
        a random sampling policy.
        
        Parameters
        ----------
        obs:
            "The observation which refers to the current state of the environment as perceived by
            the agent.
        
        Returns
        -------
        a tuple containing two values:
            action:
                a random sampled action from the possible action space.
            obs:
                the current state of the environment.
        """
        return self.test_env.action_space.sample(), obs

    def test(self, n_episodes=1, render=False):
        """
        Test a reinforcement learning policy by running a specified number of episodes and
        returning the mean reward.
        
        Parameters
        ----------
        n_episodes: int, default=1
            The number of episodes to run during testing. An episode is a complete run of 
            the environment from the initial state to a terminal state.
        render: bool, default=False
            A boolean parameter that determines whether to render the environment during testing 
            or not.
        
        Returns
        -------
        mean(episodes_rewards): float
            the mean reward obtained over the specified number of episodes.
        """
        if render:
            self.test_env.config.update({"render": 1})

        episodes_rewards = []

        for episode in range(n_episodes):
            print("Start >> Episode", episode+1)
            obs = self.test_env.reset()

            if render:
                self.test_env.render()

            steps_rewards = []
            done = False
            step = 0

            while not done:
                action, _states = self.policy(obs)
                obs, reward, done, info = self.test_env.step(action)

                if render:
                    print("Episode", episode+1, "- Step ", step+1, "- Took action: ", action,
                          "- Current State: ", obs, "- Got reward: ", reward, "- Done: ", done)
                    self.test_env.render()

                steps_rewards.append(reward)
                step += 1

            print("Done >> Episode", episode+1, "- Reward = ",
                  steps_rewards, "- Sum reward:", sum(steps_rewards))
            episodes_rewards.append(sum(steps_rewards))

        print("[INFO]  >> Mean reward: ", mean(episodes_rewards))

        self.test_env.close()

        return mean(episodes_rewards)

    def close(self):
        """
        Close the test environment.
        """
        self.test_env.close()
