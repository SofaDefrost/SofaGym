# -*- coding: utf-8 -*-
"""Release each cable of a cable-driven env from rest and check that no cable pushes.

Actions must never make a cable displacement negative (a cable can only pull); cables set negative by the scene can
only go back towards 0.

Usage:
-----
    python tests/cables_runner.py trunk-v0 8,9,10,11,12,13,14,15 --steps 5
"""

import argparse
import importlib
import os

import numpy as np


def cable_values(root):
    nodes, values = [root], []
    while nodes:
        node = nodes.pop()
        nodes.extend(node.children)
        values += [float(obj.value.value[0]) for obj in node.objects if obj.getClassName() == "CableConstraint"]
    return np.array(values)


def run(env_id, release_actions, steps):
    import gymnasium
    from gymnasium.envs.registration import load_env_creator

    import sofagym.envs  # noqa: F401, registers the environments

    importlib.import_module("splib3.animation.animate").manager = None
    env = load_env_creator(gymnasium.spec(env_id).entry_point)(None)
    failures = []
    for action in release_actions:
        env.seed(0)
        env.reset()
        floor = np.minimum(cable_values(env.root), 0.0)
        for _ in range(steps):
            env.step(action)
        values = cable_values(env.root)
        if (values < floor - 1e-9).any():
            failures.append(f"action {action}: cable values {np.round(values, 3).tolist()} below {floor.tolist()}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument("release_actions", help="Comma separated actions releasing the cables")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    failures = run(args.env_id, [int(a) for a in args.release_actions.split(",")], args.steps)
    print("\n".join(failures) if failures else "CABLES_OK", flush=True)
    # Skip SOFA's teardown.
    os._exit(0)
