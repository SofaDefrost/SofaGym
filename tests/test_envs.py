# -*- coding: utf-8 -*-
"""Every registered SofaGym env can be reset and stepped, locally, with the server architecture and with rendering.

Each env runs in a subprocess through env_runner.py.
"""

import importlib.util
import os

import pytest

from helpers import registered_env_ids, run_script

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("Sofa") is None, reason="SOFA python bindings not found, set SOFA_ROOT and PYTHONPATH"
    ),
    pytest.mark.skipif(importlib.util.find_spec("gym") is None, reason="gym is not installed"),
]


ENV_IDS = registered_env_ids()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_steps(env_id):
    run_script("env_runner.py", env_id, success_marker="ENV_OK")


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_steps_with_server(env_id):
    run_script("env_runner.py", env_id, "--server", success_marker="ENV_OK")


@pytest.mark.skipif(
    not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"), reason="rendering needs a display"
)
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_render(env_id):
    run_script("env_runner.py", env_id, "--steps", 2, "--render", success_marker="ENV_OK")
