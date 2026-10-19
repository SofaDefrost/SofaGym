# -*- coding: utf-8 -*-
"""Create a registered SofaGym env, reset it and step it with random actions.

Meant to be run in its own process (see test_envs.py).

Usage:
-----
    python tests/env_runner.py cartpole-v0 --steps 3 [--server] [--render]
"""

import argparse

import numpy as np


def run_env(env_id, steps, server, render):
    import gymnasium
    from gymnasium.envs.registration import load_env_creator

    import sofagym.envs  # registers the environments

    env_class = load_env_creator(gymnasium.spec(env_id).entry_point)
    # The env only records the positions the viewer draws when its config has render=1.
    config = {"render": 1} if render else None
    env = env_class(config, use_server=True) if server else env_class(config)
    env.seed(0)
    env.action_space.seed(0)

    try:
        observations = [env.reset()[0]]
        for _ in range(steps):
            observation, reward, done, truncated, info = env.step(env.action_space.sample())
            observations.append(observation)
            if render:
                image = np.asarray(env.render("rgb_array"))
                if image.ndim != 3 or image.std() == 0:
                    raise RuntimeError(f"Rendered image is empty: shape {image.shape}, std {image.std()}")
            if done:
                break
        if not all(np.all(np.isfinite(obs)) for obs in observations):
            raise RuntimeError(f"Non finite observation: {observations}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Id of a registered SofaGym environment, e.g. cartpole-v0")
    parser.add_argument("--steps", type=int, default=3, help="Number of env steps")
    parser.add_argument("--server", action="store_true", help="Run the simulation with the server architecture")
    parser.add_argument("--render", action="store_true", help="Render an image after each step")
    args = parser.parse_args()

    run_env(args.env_id, args.steps, args.server, args.render)
    print("ENV_OK")
