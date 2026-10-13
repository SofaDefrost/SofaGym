# -*- coding: utf-8 -*-
"""Behaviour regression, environment level: step() must behave like the SOFA v23.06 reference of each env.

The references are recorded from SofaGym f5ad4e4 on SOFA v23.06 with tests/regression/envs.py.

With a display, each episode is also rendered to tests/videos/regression/v26/<env>/<episode>.gif, and next to the
v23.06 video (recorded with the references) in tests/videos/regression/compare/<env>/<episode>.gif.
"""

import json
import os
import shutil

import pytest

from helpers import REPO_ROOT
from regression import envs

pytestmark = pytest.mark.skipif(not os.environ.get("SOFA_ROOT"), reason="SOFA_ROOT is not set")


@pytest.fixture(scope="module", autouse=True)
def remove_videos_of_removed_cases():
    for folder in ("v26", "compare"):
        for case in (envs.VIDEOS / folder).glob("*"):
            if case.is_dir() and case.name not in envs.ENVS:
                shutil.rmtree(case)


@pytest.mark.parametrize("env_id", sorted(envs.ENVS))
def test_env_behaviour_matches_v23(env_id, tmp_path):
    reference = envs.REFERENCES / f"{env_id}.json"
    if not reference.exists():
        pytest.skip(f"No reference {reference}")
    video_dir = envs.VIDEOS / "v26" / env_id if os.environ.get("DISPLAY") else None
    if video_dir:
        # Videos of episodes that no longer exist would be mistaken for current ones.
        for old in [*video_dir.glob("*.gif"), *(envs.VIDEOS / "compare" / env_id).glob("*.gif")]:
            old.unlink()
    recording = envs.record(
        env_id, tmp_path / "recording.json", os.environ["SOFA_ROOT"], REPO_ROOT, video_dir=video_dir
    )
    if video_dir:
        for video in video_dir.glob("*.gif"):
            reference_video = envs.VIDEOS / "v23" / env_id / video.name
            if reference_video.exists():
                envs.side_by_side(reference_video, video, envs.VIDEOS / "compare" / env_id / video.name)
    failures = envs.compare(json.loads(reference.read_text()), recording, env_id)
    assert not failures, "\n".join(failures)
