from abc import abstractmethod

from agents.SofaTestAgent import SofaTestAgent


class SofaBaseAgent(SofaTestAgent):
    """
    Base abstract class for creating and training reinforcement learning policies in a SOFA scene 
    with gym environment.

    Inherits from the `SofaTestAgent` class.

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
        continue until the goal is done or a termination condition happens.
    n_envs : int, default=1
        The number of environment instances created simultaneously to train the agent on 
        multiple environments in parallel.
    
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
        continue until the goal is done or a termination condition happens.
    n_envs : int
        The number of environment instances created simultaneously to train the agent on 
        multiple environments in parallel.
    test_env : gym.Env
        The test environment instance.
    env : gym.Env
        The training environment instance.
    
    Notes:
    -----
    This class is only an abstract class that needs to be inherited from and override the abstract methods defined.

    Usage:
    -----
    Create subclasses for different types of agents and inherit from this class.
    """
    def __init__(self, env_id, seed=0, output_dir="./Results", max_episode_steps=None, n_envs=1):
        """
        Initialization of base agent class. Creates the environment for the SOFA scene, 
        which will be used for the training using RL.

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
            continue until the goal is done or a termination condition happens.
        n_envs : int, default=1
            The number of environment instances created simultaneously to train the agent on 
            multiple environments in parallel.
        """
        super().__init__(env_id, seed, output_dir, max_episode_steps)
        self.n_envs = n_envs

    @abstractmethod
    def fit(self):
        """
        Train the agent using the chosen RL algorithm.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        """
        Evaluate the agent's performance after training.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls):
        """
        Load the agent from a specified checkpoint.
        """
        raise NotImplementedError
