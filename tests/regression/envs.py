# -*- coding: utf-8 -*-
"""Environment level of the behaviour regression tests: record envs with env_recorder.py and compare recordings.

Record the references (SOFA v23.06 build, SofaGym f5ad4e4 checkout, Python env with gym):
    python tests/regression/envs.py --sofa-root ~/workspace/sofa2306/build --sofagym ~/workspace/SG_main/SofaGym \
        --python ~/miniconda3/envs/sofa2306/bin/python --out tests/regression_reference/envs
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import runsofa  # noqa: E402

REFERENCES = HERE.parent / "regression_reference" / "envs"
# Videos of the episodes (not versioned): v23/ from the references, v26/ and compare/ (side by side) from the tests.
VIDEOS = HERE.parent / "videos" / "regression"

# Steps of each scripted episode and of the random episode compared by the tests: the longest prefix where v26 matched
# v23.06 (see README.md; episodes ending earlier with done are compared until then). Exceptions to the comparison,
# each with its reason:
# - tolerance: instead of TOLERANCE;
# - ignore_observations: indices of the observation not compared;
# - ignore_rewards: rewards not compared;
# - ignore_objects: prefixes of the MechanicalObject paths not compared;
# - ignore_episodes: episodes recorded (videos) but not compared.
# A case can also run a registered env (`env`) without contacts (contacts=False), replace the scripted episodes by
# physically meaningful ones (`episodes`: {name: actions}, run whole) and hold each random action (`random_hold`).
# CatheterBeam has no case: its catheter has a curved tip since the SOFA 26 update (straight on v23.06).
ENVS = {
    "bubblemotion-v0": dict(
        scene="BubbleMotion/BubbleMotionScene",
        steps=20,
        random=40,
        ignore_observations=[9, 10, 11],
        ignore_rewards=True,
        ignore_objects=["/sphere"],
        why="the ball rolling on the inflated bumps is chaotic: 1% more pressure on v26 diverges "
        "as much as v26 from v23.06; the board and cavities are compared",
    ),
    "cartpole-v0": dict(scene="CartPole/CartPoleScene", steps=20, random=40),
    "cartstem-v0": dict(
        scene="CartStem/CartStemScene",
        steps=1,
        random=5,
        why="inverted pendulum: v26 departs from v23.06 after 1 to 5 steps",
    ),
    "cartstemcontact-v0": dict(scene="CartStemContact/CartStemContactScene", steps=20, random=40),
    "catchtheobject-v0": dict(scene="CatchTheObject/CatchTheObjectScene", steps=20, random=40),
    "concentrictuberobot-v0": dict(
        scene="CTR/CTRScene",
        steps=4,
        random=14,
        tolerance=0.05,
        ignore_rewards=True,
        why="the tip of the undeployed tubes is ~0.5 mm away from v23.06 (base frame), "
        "which the reward amplifies to 10-20%; beyond ~50 mm of insertion the tube "
        "path in the cavity is bistable (each version takes either branch)",
    ),
    "concentrictuberobot-v0-no-contact": dict(
        env="concentrictuberobot-v0",
        scene="CTR/CTRScene",
        steps=20,
        random=40,
        contacts=False,
        tolerance=0.07,
        ignore_rewards=True,
        why="the contacts with the cavity are unreliable on v23.06 (the tubes go through the wall, see README.md): "
        "without them, the mechanics of the tubes (insertion, rotation, curved tips) are compared on whole "
        "episodes. On v23.06 the base of the tubes sags by ~1 mm along x (base frame), which offsets the tip "
        "by ~1 mm, 5% of its x range; the reward amplifies it",
    ),
    # Each cable (actions 2k and 2k + 1) pulled then released.
    "diamondrobot-v0": dict(
        scene="Diamond/DiamondScene",
        random=12,
        random_hold=3,
        tolerance=0.08,
        ignore_rewards=True,
        episodes={f"cable {k}": [2 * k] * 5 + [2 * k + 1] * 5 for k in range(4)},
        why="the robot starts sagging under the gravity fixed by the SOFA 26 update: 1.7 mm "
        "differently from v23.06 after the first step (7.8% of the range), below 2.5% from the "
        "third step on; the reward, a threshold on the distance to the goal, amplifies it. The "
        "random episode stops before releasing cable 2 nine times in a row (folded the robot "
        "before cables were clamped at 0)",
    ),
    # Actions: 0 up (10 mm per step), 1 down (5 mm), 2/3 +x/-x (5 mm), 4/5 +z/-z, 6/7 close/open (2.5 mm of cable, 5 on
    # v23.06). The gripper starts 13.5 mm off the cube along x and its fingertips 58 mm above its center: center,
    # lower, close and lift.
    "gripper-v0": dict(
        scene="Gripper/GripperScene",
        random=15,
        random_hold=3,
        episodes={
            "grasp": [3] * 3 + [1] * 8 + [6] * 10 + [0] * 6,
            "x": [2] * 5 + [3] * 5,
            "z": [4] * 5 + [5] * 5,
            "close and open": [6] * 5 + [7] * 5,
        },
        ignore_episodes=["grasp", "x", "z", "close and open", "random"],
        ignore_observations=list(range(24)),
        ignore_objects=["/Cube"],
        why="not comparable with v23.06, only recorded for the videos: the base of the gripper did "
        "not move (translations moved the positions of the fingers, anchored to their rest "
        "positions), the cube was a hollow FEM surface, the cables moved twice as fast",
    ),
    "maze-v0": dict(
        scene="Maze/MazeScene",
        steps=4,
        random=13,
        why="the ball rolling in the maze departs from v23.06 after 5 to 14 steps",
    ),
    "simple_maze-v0": dict(
        scene="SimpleMaze/SimpleMazeScene",
        steps=20,
        random=35,
        why="the ball departs from v23.06 after 35 random steps",
    ),
    "stempendulum-v0": dict(scene="StemPendulum/StemPendulumScene", steps=20, random=40),
    "trunk-v0": dict(
        scene="Trunk/TrunkScene",
        steps=20,
        random=40,
        ignore_observations=[63, 64, 65],
        ignore_objects=["/Goal"],
        why="the GoalMO (last 3 observations) was at the goal plus its initial position [0, -100, 100] on v23.06, "
        "it is at the goal of the reward since",
    ),
    "trunkcup-v0": dict(
        scene="TrunkCup/TrunkCupScene",
        steps=3,
        random=18,
        why="the cup grasped by friction contact departs from v23.06 after 4 to 19 steps",
    ),
}
# Steps of the reference episodes: the tests compare their first steps, the scales of the comparison come from all.
REFERENCE_STEPS, REFERENCE_RANDOM = 20, 40
# Allowed difference of observations and rewards, as a fraction of their range over all the reference episodes.
TOLERANCE = 0.02


def record(env_id, out, sofa_root, sofagym_root, python=sys.executable, video_dir=None, reference=False, timeout=1200):
    """Run the episodes of `env_id` in a new process and return the recording, or raise with the log.

    The episodes have the steps of ENVS, or REFERENCE_STEPS and REFERENCE_RANDOM for a `reference`. With `video_dir`,
    also writes a GIF of each episode there (needs a display).
    """
    sofa_root, sofagym_root, out = (pathlib.Path(p).resolve() for p in (sofa_root, sofagym_root, out))
    config = ENVS[env_id]
    episode_steps, random_steps = (
        (REFERENCE_STEPS, REFERENCE_RANDOM) if reference else (config.get("steps", REFERENCE_STEPS), config["random"])
    )
    env = dict(
        os.environ,
        SOFA_ROOT=str(sofa_root),
        RECORD_NO_CONTACT="" if config.get("contacts", True) else "1",
        RECORD_EPISODES=json.dumps(config["episodes"]) if "episodes" in config else "",
        RECORD_RANDOM_HOLD=str(config.get("random_hold", 1)),
        RECORD_GRAVITY=",".join(map(str, runsofa.GRAVITY.get(config["scene"], []))),
        # Cables only pull since the SOFA 26 update: the references of f5ad4e4 are recorded so.
        RECORD_CLAMP_CABLES="1" if reference else "",
        PYTHONPATH=os.pathsep.join([str(sofagym_root), str(sofa_root / "lib" / "python3" / "site-packages")]),
    )
    out.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [
                str(python),
                str(HERE / "env_recorder.py"),
                config.get("env", env_id),
                str(out),
                str(episode_steps),
                str(random_steps),
            ]
            + ([str(video_dir)] if video_dir else []),
            cwd=sofagym_root,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        log = result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:
        log = f"Timeout after {timeout}s\n{error.stdout or ''}{error.stderr or ''}"
    if not out.exists():
        raise RuntimeError(f"Could not record {env_id}:\n{log[-5000:]}")
    return json.loads(out.read_text())


def value_scale(values):
    """Range of each column of `values`, at least 5% of the largest range (columns that barely move)."""
    ranges = np.ptp(np.asarray(values, dtype=float).reshape(len(values), -1), axis=0)
    return np.maximum(ranges, max(0.05 * ranges.max(), 1e-9))


def compare(reference, recording, env_id):
    """Return the differences between two recordings of `env_id`, as readable strings (empty when they match).

    Only the steps of `recording` are compared, the `reference` episodes can be longer.
    """
    config = ENVS[env_id]
    tolerance = config.get("tolerance", TOLERANCE)
    episodes = reference["episodes"].values()
    observations = np.delete(
        np.array([o for e in episodes for o in e["observations"]], dtype=float),
        config.get("ignore_observations", []),
        axis=1,
    )
    scales = {
        "observations": value_scale(observations),
        "rewards": value_scale([[r] for e in episodes for r in e["rewards"]] or [[0.0]]),
    }
    keys = ["observations"] + ([] if config.get("ignore_rewards") else ["rewards"])
    failures = []
    for name, ref in reference["episodes"].items():
        if name in config.get("ignore_episodes", ()):
            continue
        new = recording["episodes"].get(name)
        if new is None:
            failures.append(f"{name}: missing episode")
            continue
        expected_steps = min(len(ref["rewards"]), new["requested_actions"])
        if len(new["rewards"]) != expected_steps:
            failures.append(f"{name}: done after {len(new['rewards'])} steps instead of {expected_steps}")
        for key in keys:
            for step, (expected, got) in enumerate(zip(ref[key], new[key])):
                if key == "observations":
                    expected, got = (
                        np.delete(np.ravel(v), config.get("ignore_observations", [])) for v in (expected, got)
                    )
                error = np.abs(np.subtract(got, expected)).ravel() / scales[key]
                if error.max() > tolerance:
                    i = int(error.argmax())
                    failures.append(
                        f"{name}: {key} at step {step} differs by {error[i]:.1%} of its range "
                        f"(index {i}): expected {np.ravel(expected)[i]:.4g}, got {np.ravel(got)[i]:.4g}"
                    )
                    break
        ignored = tuple(config.get("ignore_objects", ()))
        scene_ref = (
            {"objects": {p: o for p, o in ref["objects"].items() if not p.startswith(ignored)}} if ignored else ref
        )
        failures += [f"{name}: {failure}" for failure in runsofa.compare(scene_ref, new, config["scene"])]
    return failures


def side_by_side(reference_gif, recording_gif, out):
    """Write a GIF with the frames of `reference_gif` (left) and `recording_gif` (right), as long as the shorter one.

    The tests record only the steps they compare, the references all their steps.
    """
    import imageio

    left, right = imageio.mimread(reference_gif), imageio.mimread(recording_gif)
    frames = []
    for a, b in zip(left, right):
        a, b = a[..., :3], b[..., :3]
        height = max(a.shape[0], b.shape[0])
        a, b = (np.pad(x, ((0, height - x.shape[0]), (0, 0), (0, 0))) for x in (a, b))
        frames.append(np.hstack([a, np.full((height, 4, 3), 255, np.uint8), b]))
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=100, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sofa-root", required=True, help="SOFA build or install directory")
    parser.add_argument("--sofagym", required=True, help="SofaGym checkout whose envs are recorded")
    parser.add_argument("--python", default=sys.executable, help="Python with gym and the SofaGym requirements")
    parser.add_argument("--out", default=str(REFERENCES), help="Output directory")
    parser.add_argument("--videos", default=str(VIDEOS / "v23"), help="Directory of the videos, empty for none")
    parser.add_argument("envs", nargs="*", default=sorted(ENVS), help="Envs to record, default all")
    args = parser.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    for env_id in args.envs:
        out = pathlib.Path(args.out) / f"{env_id}.json"
        record(
            env_id,
            out,
            args.sofa_root,
            args.sofagym,
            args.python,
            video_dir=pathlib.Path(args.videos) / env_id if args.videos else None,
            reference=True,
        )
        print(f"{env_id}: {out}")
