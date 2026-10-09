# -*- coding: utf-8 -*-
"""The sofagym package can be imported, and scenes helpers do not depend on gym.

Imports run in a subprocess to start from a clean sys.modules.
"""

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def run_python(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, check=False)


def test_import_sofagym():
    result = run_python("import sofagym")
    assert result.returncode == 0, result.stderr[-5000:]


@pytest.mark.parametrize("module", ["sofagym.header", "sofagym.utils"])
def test_scene_helpers_do_not_import_gym(module):
    result = run_python(f"import sys, {module}; assert 'gym' not in sys.modules, 'gym was imported'")
    assert result.returncode == 0, result.stderr[-5000:]
