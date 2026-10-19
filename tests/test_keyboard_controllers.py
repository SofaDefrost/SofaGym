# -*- coding: utf-8 -*-
"""The keyboard controllers of the scenes react to their keys.

Each scene is run twice with its keyboard controller, from the same random seed: once without key presses, once with
the keys expected by the controller. Something must have moved between the two runs.
"""

import importlib.util
import json

import pytest

from helpers import REPO_ROOT, run_script

ENVS = REPO_ROOT / "sofagym" / "envs"

NB_PRESSES = 5

# Key codes sent by the SOFA GUI: 18 left, 19 up, 20 right, 21 down.
SCENE_KEYS = {
    "BubbleMotion": 19,  # increase the pressure of the selected cavity
    "CartStem": 20,  # push the cart right
    "CartStemContact": 20,  # push the cart right
    "CatchTheObject": 20,  # increase the pressure
    "StemPendulum": 20,  # apply a torque
    "Gripper": ord("1"),  # pull the finger cables
}

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("Sofa") is None, reason="SOFA python bindings not found, set SOFA_ROOT and PYTHONPATH"
)


def run_scene(scene_file, dump, keys=()):
    key_args = [arg for key in keys for arg in ("--key", key)]
    run_script(
        "scene_runner.py", scene_file, "--keyboard-controller", "--dump", dump, *key_args, success_marker="SCENE_OK"
    )
    return json.loads(dump.read_text())


@pytest.mark.parametrize("env_name", SCENE_KEYS)
def test_keyboard_controller_moves_scene(env_name, tmp_path):
    scene_file = ENVS / env_name / f"{env_name}Scene.py"

    without_keys = run_scene(scene_file, tmp_path / "without_keys.json")
    with_keys = run_scene(scene_file, tmp_path / "with_keys.json", [SCENE_KEYS[env_name]] * NB_PRESSES)

    assert without_keys.keys() == with_keys.keys()
    moved = [path for path in with_keys if with_keys[path] != without_keys[path]]
    assert moved, f"Pressing key {SCENE_KEYS[env_name]} did not move anything in {env_name}"
