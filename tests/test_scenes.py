# -*- coding: utf-8 -*-
"""Every SOFA scene in sofagym/envs can be initialized and animated.

Each scene runs in a subprocess through scene_runner.py.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER = pathlib.Path(__file__).resolve().parent / "scene_runner.py"
SCENE_FILES = sorted((REPO_ROOT / "sofagym" / "envs").glob("*/*Scene*.py"))

NB_STEPS = 10
TIMEOUT = 300

pytestmark = pytest.mark.skipif(importlib.util.find_spec("Sofa") is None,
                                reason="SOFA python bindings not found, set SOFA_ROOT and PYTHONPATH")


@pytest.mark.parametrize("scene_file", SCENE_FILES, ids=lambda path: path.stem)
def test_scene_init_and_animate(scene_file):
    # Scenes import sofagym.*, make the repo importable without installing the package.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))

    result = subprocess.run([sys.executable, str(RUNNER), str(scene_file), "--steps", str(NB_STEPS)],
                            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=TIMEOUT,
                            check=False)

    output = (result.stdout + result.stderr)[-5000:]
    assert result.returncode == 0 and "SCENE_OK" in result.stdout, \
        f"{scene_file.relative_to(REPO_ROOT)} failed (exit code {result.returncode}):\n{output}"
