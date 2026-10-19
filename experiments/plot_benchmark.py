# -*- coding: utf-8 -*-
"""Plot the RL benchmark (benchmark.py): reward of the tests against the training steps, mean and 95% CI over seeds
(normal approximation: 1.96 standard errors).

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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
        # The seeds of an algorithm test at the same steps.
        steps, rewards = data.setdefault(match["env"], {}).setdefault(match["algo"], (values["steps"], []))
        rewards.append(values["final_rewards" if args.final else "rewards"][: len(steps)])

    kind = "final_rewards" if args.final else "rewards"
    for env_id, algos in data.items():
        plt.figure()
        for algo, (steps, rewards) in sorted(algos.items()):
            rewards = np.array(rewards)
            mean = rewards.mean(axis=0)
            ci = 1.96 * rewards.std(axis=0, ddof=1) / np.sqrt(len(rewards)) if len(rewards) > 1 else 0 * mean
            plt.plot(steps, mean, label=algo)
            plt.fill_between(steps, mean - ci, mean + ci, alpha=0.2)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.legend(title="Algorithm")
        plt.title(f"{env_id} ({kind.replace('_', ' ')} of the tests)")
        out = pathlib.Path(args.results) / f"{env_id}_{kind}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(out)
