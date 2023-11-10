import os
from typing import Optional

import gym
from colorama import Fore
from stable_baselines3.common.utils import set_random_seed


def make_env(env_id: str, rank: int = 0, seed: int = 0, max_episode_steps: Optional[int] = None, config: Optional[dict] = None):
    """Utility function for creating gym envs.

    Parameters
    ----------
    env_id: str
        The environment ID
    rank: int, default=0
        Index of the subprocess
    seed: int, default=0
        The inital seed for RNG
    max_episode_steps: int, default=None
        Maximum number of steps to perform per episode.
    config: dict, default=None
        The configuration parameters for the environment.

    Returns
    -------
    Callable
        The callable function to create the environment.
    """
    def _init():
        env_seed = seed + rank
        env_config = config
        if env_config is None:
            env_config = {"seed": env_seed}
        else:
            env_config['seed'] = env_seed

        env = gym.make(env_id, config=env_config)
        env.seed(env_seed)
        env.reset()

        if max_episode_steps:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        return env
    
    set_random_seed(seed)

    return _init

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    return d

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def args_check(arg, valid_list, arg_type):
    valid_arg_values = valid_list.values()
    if arg not in valid_arg_values:
        print(Fore.RED + '[ERROR]   ' + Fore.RESET + f"Invalid {arg_type}.")
        raise SystemExit(f"Valid {arg_type} options:\n{[value for value in valid_arg_values]}")
