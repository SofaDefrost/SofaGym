# -*- coding: utf-8 -*-
"""Every scene declares its UNITS (Sofa.Units.MechanicalUnitSystem), consistent with its gravity.

Each scene runs in a subprocess through scene_runner.py.
"""

import importlib.util

import pytest

from helpers import REPO_ROOT, run_script

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("Sofa") is None, reason="SOFA python bindings not found, set SOFA_ROOT and PYTHONPATH"
)


@pytest.mark.parametrize(
    "scene_file", sorted((REPO_ROOT / "sofagym" / "envs").glob("*/*Scene*.py")), ids=lambda scene_file: scene_file.stem
)
def test_scene_units_match_gravity(scene_file):
    run_script("scene_runner.py", scene_file, "--steps", 1, "--check-units", success_marker="SCENE_OK")
