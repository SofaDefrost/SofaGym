# -*- coding: utf-8 -*-
"""Train and run agents on a registered SofaGym env, or check its planning features.

Meant to be run in its own process (see test_rl.py). Prints RL_OK on success, RL_SKIP <reason> when the check cannot
run here.

Usage:
-----
    python tests/rl_runner.py sb3 cartpole-v0 on-policy --out /tmp/rl
    python tests/rl_runner.py rlberry cartpole-v0 --out /tmp/rl
    python tests/rl_runner.py planning cartpole-v0
    python tests/rl_runner.py history cartpole-v0
"""

import argparse
import importlib
import os
import sys

import numpy as np

# Tiny trainings: they check that training and inference run, not that the agent learns.
EPISODE_STEPS = 4
TIMESTEPS = 16
SB3_OVERRIDES = {
    "PPO": dict(n_steps=8, batch_size=8, n_epochs=1),
    "A2C": dict(n_steps=8),
    "DQN": dict(buffer_size=1000, learning_starts=4, batch_size=8),
    "SAC": dict(buffer_size=1000, learning_starts=4, batch_size=8),
}
SMALL_NET = {
    "PPO": "dict(net_arch=dict(pi=[16], vf=[16]))",
    "A2C": "dict(net_arch=dict(pi=[16], vf=[16]))",
    "DQN": "dict(net_arch=[16])",
    "SAC": "dict(net_arch=dict(pi=[16], qf=[16]))",
}


def env_class(env_id):
    import gym
    from gym.envs.registration import load

    import sofagym.envs  # noqa: F401, registers the environments

    return load(gym.envs.registry[env_id].entry_point)


def new_env(env_id, config=None, **kwargs):
    """A fresh env, seeded, with the splib3 animation manager reset as for any new scene."""
    importlib.import_module("splib3.animation.animate").manager = None
    env = env_class(env_id)(config, **kwargs)
    env.seed(0)
    return env


def two_actions(env):
    """Two different actions: the first two of a discrete space, the bounds of a continuous one."""
    from gym import spaces

    space = env.action_space
    if isinstance(space, spaces.Discrete):
        return 0, 1
    return space.low.tolist(), space.high.tolist()


def sb3(env_id, kind, out):
    """Train an SB3Agent, reload it from its checkpoints, and evaluate it."""
    from gym import spaces

    env = new_env(env_id)
    discrete = isinstance(env.action_space, spaces.Discrete)
    env.close()
    algo = "PPO" if kind == "on-policy" else ("DQN" if discrete else "SAC")

    import yaml

    from agents.SB3Agent import SB3Agent

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, "agents", "hyperparameters", "stable_baselines_params.yml")) as f:
        params = yaml.safe_load(f)[algo]
    params["init_kwargs"].update(SB3_OVERRIDES[algo], policy_kwargs=SMALL_NET[algo])
    params["fit_kwargs"].update(
        total_timesteps=TIMESTEPS,
        max_episode_steps=EPISODE_STEPS,
        eval_freq=TIMESTEPS // 2,
        n_eval_episodes=1,
        save_freq=TIMESTEPS // 2,
        video_length=EPISODE_STEPS,
    )

    # As rl.py: results in ./Results, relative to the working directory.
    os.chdir(out)
    agent = SB3Agent(env_id, algo, 0, "./Results", EPISODE_STEPS, 1, **params)
    agent.fit(TIMESTEPS)
    model_dir = agent.model_dir
    agent.close()

    loaded = SB3Agent.load(model_dir)
    loaded.eval(1, model_timestep="best_model", render=False, record=False)
    loaded.close()


def rlberry(env_id, out):
    """Train an RLberryAgent, reload it and evaluate it."""
    try:
        from agents.RLberryAgent import RLberryAgent
    except Exception as error:  # rlberry missing, or incompatible with the installed gym/gymnasium
        print(f"RL_SKIP RLberryAgent cannot be imported: {error!r}", flush=True)
        os._exit(0)

    os.chdir(out)
    agent = RLberryAgent(env_id, "PPO", 0, "./Results", EPISODE_STEPS, 1)
    agent.fit(1)
    model_dir = agent.model_dir
    loaded = RLberryAgent.load(model_dir)
    loaded.eval(1)


def same(name, observation, expected, failures):
    observation, expected = np.asarray(observation, dtype=float), np.asarray(expected, dtype=float)
    if observation.shape != expected.shape or not np.allclose(observation, expected, rtol=1e-5, atol=1e-6):
        failures.append(f"{name}: {np.round(observation, 4).tolist()} != {np.round(expected, 4).tolist()}")


def planning(env_id):
    """Branch with the server in planning mode, the results must match local runs of the same action sequences."""
    env = new_env(env_id, {"planning": True}, use_server=True)
    a, b = two_actions(env)
    failures = []
    try:
        env.reset()
        o_a = env.step(a)[0]
        history_a = list(env.past_actions)
        env.past_actions = []
        o_b = env.step(b)[0]
        # Expand a after exploring b.
        env.past_actions = history_a
        o_ab = env.step(b)[0]
        env.past_actions = []
        o_a_again = env.step(a)[0]
        # The agent acts a: the other branches are closed, the subtree of a is kept.
        env.past_actions = history_a
        env.clean()
        o_ab_again = env.step(b)[0]
    finally:
        env.close()

    expected = {}
    for actions in ([a, b], [b]):
        local = new_env(env_id)
        local.reset()
        expected[str(actions)] = [local.step(action)[0] for action in actions]
        local.close()

    same("step a", o_a, expected[str([a, b])][0], failures)
    same("step b from the root (branch)", o_b, expected[str([b])][0], failures)
    same("step a again from the root", o_a_again, o_a, failures)
    same("step b after a", o_ab, expected[str([a, b])][1], failures)
    same("step b after a, once a is acted", o_ab_again, o_ab, failures)
    return failures


def history(env_id):
    """An env started from an action history resets to the state reached by stepping these actions."""
    local = new_env(env_id)
    a, _ = two_actions(local)
    local.reset()
    o_a = local.step(a)[0]
    formatted = local._formataction(a)
    local.close()

    env = new_env(env_id, {"start_from_history": [formatted]})
    obs = env.reset()
    env.close()
    failures = []
    same("reset after history [a]", obs, o_a, failures)
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["sb3", "rlberry", "planning", "history"])
    parser.add_argument("env_id", help="Id of a registered SofaGym environment, e.g. cartpole-v0")
    parser.add_argument("kind", nargs="?", choices=["on-policy", "off-policy"], help="sb3: kind of algorithm")
    parser.add_argument("--out", help="sb3, rlberry: directory of the results")
    args = parser.parse_args()

    if args.check == "sb3":
        sb3(args.env_id, args.kind, args.out)
        failures = []
    elif args.check == "rlberry":
        rlberry(args.env_id, args.out)
        failures = []
    else:
        failures = globals()[args.check](args.env_id)
    print("\n".join(failures) if failures else "RL_OK", flush=True)
    sys.stdout.flush()
    # Skip SOFA's teardown and the threads of the server.
    os._exit(0)
