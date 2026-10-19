# -*- coding: utf-8 -*-
"""Plot the RL benchmark (benchmark.py): reward of the tests against the training steps, mean and 95% CI over seeds.

As the original plot.py of SofaGym (2022), one figure per env, one curve per algorithm.

Usage:
-----
    python experiments/plot_benchmark.py [experiments/results/benchmark] [--final]
"""

import argparse
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

RUN = re.compile(r"(?P<algo>[A-Z]+)_(?P<env>.+)_(?P<seed>\d+)$")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results", nargs="?", default="experiments/results/benchmark", help="Directory of the runs of benchmark.py"
    )
    parser.add_argument("--final", action="store_true", help="Final rewards of the test episodes, not total rewards")
    args = parser.parse_args()

    data = {}
    for log in sorted(pathlib.Path(args.results).glob("*/log.json")):
        match = RUN.match(log.parent.name)
        if not match:
            continue
        values = json.loads(log.read_text())
        rows = data.setdefault(match["env"], {"Steps": [], "Reward": [], "Algorithm": []})
        rows["Steps"] += values["steps"]
        rows["Reward"] += values["final_rewards" if args.final else "rewards"]
        rows["Algorithm"] += [match["algo"]] * len(values["steps"])

    kind = "final_rewards" if args.final else "rewards"
    for env_id, rows in data.items():
        plt.figure()
        sns.lineplot(data=rows, x="Steps", y="Reward", hue="Algorithm")
        plt.title(f"{env_id} ({kind.replace('_', ' ')} of the tests)")
        out = pathlib.Path(args.results) / f"{env_id}_{kind}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(out)
