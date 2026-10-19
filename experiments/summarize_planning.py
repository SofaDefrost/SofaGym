# -*- coding: utf-8 -*-
"""Summarize the planning experiments (planning.py): per env, targets reached and mean distance, steps and time.

Usage:
-----
    python experiments/summarize_planning.py experiments/results/planning
"""

import json
import pathlib
import sys

import numpy as np

if __name__ == "__main__":
    results = pathlib.Path(sys.argv[1])
    by_env = {}
    for path in sorted(results.glob("*.json")):
        result = json.loads(path.read_text())
        by_env.setdefault(result["env_id"], []).append(result)

    lines = ["| Env | Targets reached | Mean distance | Mean steps | Mean time (s) |", "|---|---|---|---|---|"]
    for env_id, runs in by_env.items():
        reached = sum(run["reached"] for run in runs)
        lines.append(
            f"| {env_id} | {reached}/{len(runs)} | {np.mean([r['distance'] for r in runs]):.2f} "
            f"| {np.mean([r['steps'] for r in runs]):.1f} | {np.mean([r['time'] for r in runs]):.1f} |"
        )
    summary = "\n".join(lines)
    (results / "summary.md").write_text(summary + "\n")
    print(summary)
