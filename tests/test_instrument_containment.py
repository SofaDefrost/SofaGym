# -*- coding: utf-8 -*-
"""Pushed instruments stay inside the cavity (CTR) or the vessels (CatheterBeam): otherwise the contacts broke.

Each instrument is inserted alone: 15 steps in CTR's cavity (150, 120 and 100 mm), 450 mm in CatheterBeam's vessels;
each env runs in a subprocess (containment_runner.py).
With a display, the pushes are rendered to tests/videos/containment/<env>/instrument_<i>.gif.
"""

import importlib.util
import os

import pytest

from helpers import REPO_ROOT, run_script

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("Sofa") is None, reason="SOFA python bindings not found, set SOFA_ROOT and PYTHONPATH"
)


@pytest.mark.parametrize(
    "env_id, length", [("concentrictuberobot-v0", ["--steps", 15]), ("catheter_beam-v0", ["--insertion", 450])]
)
def test_pushed_instruments_stay_inside(env_id, length):
    video = []
    if os.environ.get("DISPLAY"):
        video_dir = REPO_ROOT / "tests" / "videos" / "containment" / env_id
        for old in video_dir.glob("*.gif"):
            old.unlink()
        video = ["--video", video_dir]
    run_script("containment_runner.py", env_id, *length, *video, success_marker="CONTAINED_OK")
