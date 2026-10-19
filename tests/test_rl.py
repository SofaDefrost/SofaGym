# -*- coding: utf-8 -*-
"""Training and inference with the agents of agents/, and the planning features of the server, on every env.

- SB3: a tiny training with SB3Agent (on-policy PPO, off-policy DQN or SAC), then the agent is reloaded from its
  checkpoints and its best model evaluated.
- Planning: the server in planning mode branches between action sequences, and must give the same observations as
  local runs of these sequences; rl-agents' tree search planners plan with it (discrete actions); an env started from
  an action history must reset to the state they reach.

Each check runs in a subprocess through rl_runner.py.
"""

import importlib.util

import pytest

from helpers import registered_env_ids, run_script

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("Sofa") is None, reason="SOFA python bindings not found, set SOFA_ROOT and PYTHONPATH"
    ),
    pytest.mark.skipif(importlib.util.find_spec("gymnasium") is None, reason="gymnasium is not installed"),
]

ENV_IDS = registered_env_ids()


@pytest.mark.parametrize("kind", ["on-policy", "off-policy"])
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_sb3_train_and_infer(env_id, kind, tmp_path):
    run_script("rl_runner.py", "sb3", env_id, kind, "--out", tmp_path, success_marker="RL_OK")


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_planning_branches(env_id):
    run_script("rl_runner.py", "planning", env_id, success_marker="RL_OK")


@pytest.mark.skipif(importlib.util.find_spec("rl_agents") is None, reason="rl-agents is not installed")
@pytest.mark.parametrize("planner", ["deterministic", "mcts"])
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_rl_agents_planner(env_id, planner):
    output = run_script("rl_runner.py", "planner", env_id, planner, success_marker="RL_")
    if "RL_SKIP" in output:
        pytest.skip(output.split("RL_SKIP", 1)[1].strip())
    assert "RL_OK" in output


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_start_from_history(env_id):
    run_script("rl_runner.py", "history", env_id, success_marker="RL_OK")
