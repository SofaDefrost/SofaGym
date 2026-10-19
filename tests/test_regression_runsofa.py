# -*- coding: utf-8 -*-
"""Behaviour regression, runSofa level: every scene must move like its SOFA v23.06 reference.

The references are recorded from SofaGym f5ad4e4 on SOFA v23.06 with tests/regression/runsofa.py.
"""

import json
import os
import pathlib

import pytest

from helpers import REPO_ROOT
from regression import runsofa

pytestmark = pytest.mark.skipif(not os.environ.get("SOFA_ROOT"), reason="SOFA_ROOT is not set")


@pytest.mark.parametrize("scene", sorted(runsofa.STEPS), ids=lambda scene: pathlib.Path(scene).name)
def test_scene_behaviour_matches_v23(scene, tmp_path):
    reference = runsofa.REFERENCES / f"{pathlib.Path(scene).name}.json"
    if not reference.exists():
        pytest.skip(f"No reference {reference}")
    recording = runsofa.record(scene, tmp_path / "recording.json", os.environ["SOFA_ROOT"], REPO_ROOT)
    failures = runsofa.compare(json.loads(reference.read_text()), recording, scene)
    assert not failures, "\n".join(failures)
