# -*- coding: utf-8 -*-
"""Cables can only pull: releasing a cable from rest never makes it push (cables_runner.py, one subprocess per env)."""

import importlib.util

import pytest

from helpers import run_script

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("Sofa") is None, reason="SOFA python bindings not found, set SOFA_ROOT and PYTHONPATH"
)

# Env id: actions releasing a cable.
RELEASE_ACTIONS = {
    "diamondrobot-v0": "1,3,5,7",
    "trunk-v0": "8,9,10,11,12,13,14,15",
    "trunkcup-v0": "8,9,10,11,12,13,14,15",
    "gripper-v0": "7",
}


@pytest.mark.parametrize("env_id", sorted(RELEASE_ACTIONS))
def test_released_cables_do_not_push(env_id):
    run_script("cables_runner.py", env_id, RELEASE_ACTIONS[env_id], success_marker="CABLES_OK")
