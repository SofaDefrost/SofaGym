# -*- coding: utf-8 -*-
"""Build a SofaGym scene, initialize it and animate it for a few steps.

Meant to be run in its own process (see test_scenes.py): scenes of different
environments share module names (Maze.py, Sphere.py, ...) and SOFA may crash
hard, so each scene gets a fresh interpreter.

Usage:
-----
    python tests/scene_runner.py sofagym/envs/CartPole/CartPoleScene.py --steps 10
    python tests/scene_runner.py sofagym/envs/CartStem/CartStemScene.py --keyboard-controller --key 20 --dump out.json
"""

import argparse
import ast
import copy
import importlib.util
import inspect
import json
import math
import os
import pathlib
import random
import sys


def load_default_config(env_file):
    """Return the DEFAULT_CONFIG of the env class defined in env_file.

    The class body is evaluated from source instead of importing the module, so
    that the scenes can be tested without the gym dependency.
    """
    tree = ast.parse(pathlib.Path(env_file).read_text(), filename=str(env_file))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        assignments = [stmt for stmt in node.body if isinstance(stmt, ast.Assign)]
        if not any(target.id == "DEFAULT_CONFIG" for stmt in assignments for target in stmt.targets
                   if isinstance(target, ast.Name)):
            continue
        namespace = {"os": os, "sys": sys, "__file__": str(env_file)}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(env_file), "exec"), namespace)
        return namespace["DEFAULT_CONFIG"]
    raise LookupError(f"No DEFAULT_CONFIG found in {env_file}")


def make_config(scene_file, scene_defaults=None):
    """Mimic the config AbstractEnv hands to createScene, with a deterministic goal.

    scene_defaults (the createScene default config) fill keys that envs only set at runtime, e.g. init_pos.
    """
    env_dir = pathlib.Path(scene_file).parent
    config = copy.deepcopy(scene_defaults or {})
    config.update(copy.deepcopy(load_default_config(env_dir / f"{env_dir.name}Env.py")))
    config["dt"] = config.get("dt", 0.01)
    if config.get("goalList"):
        config.update({"goal_node": 0, "goalPos": config["goalList"][0]})
    return config


def load_scene_module(scene_file):
    spec = importlib.util.spec_from_file_location(pathlib.Path(scene_file).stem, scene_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def press_keys(root, keys):
    """Send key presses to every controller of the scene, as the GUI does."""
    nb_controllers = 0
    nodes = [root]
    while nodes:
        node = nodes.pop()
        nodes.extend(node.children)
        for obj in node.objects:
            if hasattr(obj, "onKeypressedEvent"):
                nb_controllers += 1
                for key in keys:
                    obj.onKeypressedEvent({"key": key})
    if nb_controllers == 0:
        raise RuntimeError("No keyboard controller found in the scene")


def mechanical_positions(root):
    positions = {}
    nodes = [root]
    while nodes:
        node = nodes.pop()
        nodes.extend(node.children)
        for obj in node.objects:
            if obj.getClassName() == "MechanicalObject":
                positions[obj.getPathName()] = obj.position.value.tolist()
    return positions


def run_scene(scene_file, steps, keyboard_controller=False, keys=(), dump=None):
    import numpy as np
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
    import SofaRuntime

    # Same seed in every run, so that two runs only differ by the key presses.
    random.seed(0)
    np.random.seed(0)

    module = load_scene_module(scene_file)

    root = Sofa.Core.Node("root")
    SofaRuntime.importPlugin("Sofa.Component")

    parameters = inspect.signature(module.createScene).parameters
    if "config" in parameters:
        scene_defaults = parameters["config"].default
        if scene_defaults is inspect.Parameter.empty:
            scene_defaults = None
        config = make_config(scene_file, scene_defaults)
        config["keyboard_controller"] = keyboard_controller
        module.createScene(root, config, mode="simu_and_visu")
    else:
        module.createScene(root)
    Sofa.Simulation.init(root)

    dt = root.dt.value
    for step in range(steps):
        if keys and step == steps // 2:
            press_keys(root, keys)
        Sofa.Simulation.animate(root, dt)

    expected_time = steps * dt
    if not math.isclose(root.time.value, expected_time, rel_tol=1e-6):
        raise RuntimeError(f"Simulation time is {root.time.value}, expected {expected_time}")

    if dump:
        pathlib.Path(dump).write_text(json.dumps(mechanical_positions(root)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_file", help="Path to a *Scene*.py file defining createScene")
    parser.add_argument("--steps", type=int, default=10, help="Number of animation steps")
    parser.add_argument("--keyboard-controller", action="store_true", help="Add the scene keyboard controller")
    parser.add_argument("--key", type=int, action="append", default=[],
                        help="Key code to press halfway through the simulation, can be repeated")
    parser.add_argument("--dump", help="Write the positions of all MechanicalObjects to this json file")
    args = parser.parse_args()

    run_scene(os.path.abspath(args.scene_file), args.steps, args.keyboard_controller,
              [chr(key) for key in args.key], args.dump)
    print("SCENE_OK")
