import os
from typing import Optional

import gym
import gymnasium
import shimmy
from colorama import Fore
from stable_baselines3.common.utils import set_random_seed


def make_env(
    env_id: str, rank: int = 0, seed: int = 0, max_episode_steps: Optional[int] = None, config: Optional[dict] = None
):
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
        # The subprocesses of SubprocVecEnv start fresh: register the environments there.
        import sofagym.envs  # noqa: F401

        env_seed = seed + rank
        env_config = config
        if env_config is None:
            env_config = {"seed": env_seed}
        else:
            env_config["seed"] = env_seed

        # gym 0.26's env checker uses np.bool8, removed in NumPy 2.
        env = gym.make(env_id, config=env_config, disable_env_checker=True)
        env.seed(env_seed)
        env.reset()

        # SB3 >= 2 works with gymnasium envs: adapt the gym 0.21 API of the SofaGym envs, with the render mode of the
        # viewer (SB3's own adapter sets none, which its video recorder refuses).
        env = shimmy.GymV21CompatibilityV0(env=env, render_mode="rgb_array" if env_config.get("render") else None)
        if max_episode_steps:
            env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        return env

    set_random_seed(seed)

    return _init


def sec_to_hours(seconds):
    a = str(seconds // 3600)
    b = str((seconds % 3600) // 60)
    c = str((seconds % 3600) % 60)
    d = ["{} hours {} mins {} seconds".format(a, b, c)]
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
        print(Fore.RED + "[ERROR]   " + Fore.RESET + f"Invalid {arg_type}.")
        raise SystemExit(f"Valid {arg_type} options:\n{[value for value in valid_arg_values]}")
