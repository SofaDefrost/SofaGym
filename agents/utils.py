import os
from typing import Optional

import gym
from colorama import Fore
from stable_baselines3.common.utils import set_random_seed


def make_env(env_id: str, rank: int = 0, seed: int = 0, max_episode_steps: Optional[int] = None, config: Optional[dict] = None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    :param max_episode_steps: maximum number of steps to perform per episode
    """
    def _init():
        env = gym.make(env_id, config=config)
        env.seed(seed + rank)
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
