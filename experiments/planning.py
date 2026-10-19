# -*- coding: utf-8 -*-
"""Planning experiments of the SofaGym paper (sections 5.3 and 5.4): OPD solving one target of an env.

OPD (Optimistic Planning for Deterministic systems, Hren & Munos 2008, rl-agents' DeterministicPlannerAgent) plans with
the server in planning mode: each simulated transition is a forked SOFA client. After each real action, the agent closes
the clients of the branches it left (env.clean()). The episode ends when the target is reached (terminated) or after
timer_limit steps. The actions are then replayed in a local env to measure the final distance to the target.

Targets: random ones (Diamond, ConcentricTubeRobot, drawn from the seed), or the goals of the env (Trunk's 2 goals,
SimpleMaze's 4 corners).

Usage:
-----
    python experiments/planning.py diamondrobot-v0 0 --budget 100 --out experiments/results/planning
"""

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# env id: goals of the env to use as targets, or None for random targets drawn from the seed.
ENVS = {
    "diamondrobot-v0": None,
    "concentrictuberobot-v0": None,
    "trunk-v0": [[40, 40, 100], [-10, 20, 80]],
    "simple_maze-v0": [301, 334, 317, 312],
}
GAMMA = 0.9


def make(env_id, target, planning, render_mode=None):
    import gymnasium

    import sofagym.envs  # noqa: F401, registers the environments

    goals = ENVS[env_id]
    config = {"goalList": [goals[target]]} if goals else {}
    if planning:
        return gymnasium.make(env_id, config={**config, "planning": True}, use_server=True)
    return gymnasium.make(env_id, config=config, render_mode=render_mode)


def run(env_id, target, budget, out):
    from rl_agents.agents.tree_search.deterministic import DeterministicPlannerAgent

    env = make(env_id, target, planning=True)
    agent = DeterministicPlannerAgent(env, {"budget": budget, "gamma": GAMMA})
    # OPD breaks the ties between actions at random.
    agent.seed(target)
    obs, _ = env.reset(seed=target)
    goal = env.unwrapped.config["goalPos"]
    actions, rewards, terminated, truncated = [], [], False, False
    start = time.time()
    while not (terminated or truncated):
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.unwrapped.clean()
        actions.append(int(action))
        rewards.append(float(reward))
        print(f"step {len(actions)}: action {action}, reward {reward:.3f}, {time.time() - start:.0f} s", flush=True)
    duration = time.time() - start
    env.close()

    # The final distance to the target, by the reward of the scene, replaying the actions locally.
    local = make(env_id, target, planning=False)
    local.reset(seed=target)
    for action in actions:
        local.step(action)
    distance = float(local.unwrapped.root.Reward.getReward()[1])
    local.close()

    result = dict(
        env_id=env_id,
        target=target,
        goal=goal,
        budget=budget,
        gamma=GAMMA,
        reached=bool(terminated),
        distance=distance,
        steps=len(actions),
        time=duration,
        actions=actions,
        rewards=rewards,
    )
    path = pathlib.Path(out) / f"{env_id}_{target}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k not in ("actions", "rewards")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env_id", choices=sorted(ENVS))
    parser.add_argument("target", type=int, help="Seed of a random target, or index of a goal of the env")
    parser.add_argument("--budget", type=int, default=100, help="Simulated transitions per real step")
    parser.add_argument("--out", default="experiments/results/planning")
    args = parser.parse_args()
    # The threads of the server keep the process alive after an exception: exit explicitly, skipping SOFA's teardown.
    try:
        run(args.env_id, args.target, args.budget, args.out)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    print("PLANNING_DONE", flush=True)
    os._exit(0)
