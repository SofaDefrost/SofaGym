# -*- coding: utf-8 -*-
"""Every SOFA scene in sofagym/envs can be initialized and animated.

Each scene runs in a subprocess through scene_runner.py.
"""

import importlib.util

import pytest

from helpers import REPO_ROOT, run_script

SCENE_FILES = sorted((REPO_ROOT / "sofagym" / "envs").glob("*/*Scene*.py"))

NB_STEPS = 10

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("Sofa") is None, reason="SOFA python bindings not found, set SOFA_ROOT and PYTHONPATH"
)


@pytest.mark.parametrize("scene_file", SCENE_FILES, ids=lambda path: path.stem)
def test_scene_init_and_animate(scene_file):
    run_script("scene_runner.py", scene_file, "--steps", NB_STEPS, success_marker="SCENE_OK")
