# -*- coding: utf-8 -*-
"""Run a test script in a subprocess with sofagym importable, and check its outcome."""

import os
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = pathlib.Path(__file__).resolve().parent
TIMEOUT = 300

# SOFA reports many failures (e.g. exceptions in Python controllers) as log messages without raising.
SOFA_ERROR = re.compile(r"^\[(ERROR|FATAL)\]", re.MULTILINE)


def run_script(script, *args, success_marker):
    """Run tests/<script> with args, assert it succeeded without SOFA errors, return its stdout."""
    env = dict(os.environ)
    # Scripts import sofagym.*, make the repo importable without installing the package.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))

    result = subprocess.run(
        [sys.executable, str(TESTS_DIR / script), *map(str, args)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )

    output = result.stdout + result.stderr
    errors = sorted(set(line for line in output.splitlines() if SOFA_ERROR.match(line)))
    assert result.returncode == 0 and success_marker in result.stdout and not errors, (
        f"{script} {' '.join(map(str, args))} failed (exit code {result.returncode})\n"
        f"SOFA errors:\n" + "\n".join(errors[:20]) + f"\n\nOutput tail:\n{output[-5000:]}"
    )
    return result.stdout
