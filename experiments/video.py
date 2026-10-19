# -*- coding: utf-8 -*-
"""Videos of the experiments (needs a display): one episode of the best trained agent of each algorithm and env, and
the plan of OPD for each target.

The trained agent is the best model (best.zip) of the seed with the best last test; its episode uses the deterministic
policy. A video is made again only when its model or plan is newer.

Usage:
-----
    python experiments/video.py experiments/results
"""

import json
import multiprocessing
import os
import pathlib
import sys

import numpy as np
import Sofa.Core
from Sofa.Types import RGBAColor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from planning import make  # noqa: E402

EPISODE_SEED = 1000  # not a training seed
# Planning env id: its goal and its effector (the point the reward brings to the goal) from the root of the scene, the
# radius of the goal sphere (the distance at which the target is reached) and of the effector sphere (None: not drawn).
MARKERS = {
    "diamondrobot-v0": dict(
        positions=lambda root: (root.goal.goalMO.position.value[0], root.Robot.Actuators.actuatedPoints.position.value[0]),
        goal=5.0,
        effector=1.5,
    ),
    "concentrictuberobot-v0": dict(
        positions=lambda root: (root.Goal.GoalMO.position.value[0], root.InstrumentCombined.DOFs.position.value[-1][:3]),
        goal=3.0,
        effector=None,
    ),
    # Reached at 1, too small to see: 3.
    "trunk-v0": dict(
        positions=lambda root: (root.Goal.GoalMO.position.value[0], root.Reward._computeTips()), goal=3.0, effector=1.5
    ),
    "simple_maze-v0": dict(
        positions=lambda root: (root.Goal.GoalMO.position.value[0], root.Sphere.sphere_mo.position.value[0]),
        goal=10.0,
        effector=3.0,
    ),
}


class Markers(Sofa.Core.Controller):
    """Draws the goal (translucent green sphere) and the effector (red sphere) in the scene, keeps their distance."""

    def __init__(self, root, positions, goal, effector, **kwargs):
        Sofa.Core.Controller.__init__(self, **kwargs)
        self.root, self.positions, self.goal, self.effector = root, positions, goal, effector
        self.distance = None

    def draw(self, vparams):
        goal, effector = (np.asarray(p, dtype=float)[None] for p in self.positions(self.root))
        self.distance = float(np.linalg.norm(goal - effector))
        draw = vparams.getDrawTool()
        draw.drawSpheres(goal, [self.goal], RGBAColor(0, 1, 0, 0.4))
        if self.effector:
            draw.drawSpheres(effector, [self.effector], RGBAColor(1, 0, 0, 1))


def add_markers(root, markers):
    """Add the markers to the scene, hide the goal the scene draws itself (a fixed size, not the reached distance)."""
    for node in root.children:
        if node.name.value.lower() == "goal":
            for obj in node.objects:
                if obj.findData("showObject"):
                    obj.showObject = False
    root.addObject(Markers(root, name="VideoMarkers", **markers))


def frame(env):
    """The rendered scene, and the distance from the effector to the goal when it has markers."""
    image = np.asarray(env.render())
    # The window of the viewer has the scene on top of an agent panel.
    image = np.ascontiguousarray(image[: env.unwrapped.viewer.surface_size[1], :, :3], dtype=np.uint8)
    markers = env.unwrapped.root.getObject("VideoMarkers")
    return image, markers and f"distance {markers.distance:.2f}"


def crop(frames, margin=40):
    """Crop the frames to what is drawn in any of them, downsample to at most 800 pixels wide."""
    rows, cols = np.nonzero(np.any(np.stack(frames) > 0, axis=(0, 3)))
    top, left = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
    bottom, right = rows.max() + margin, cols.max() + margin
    step = max(1, -(-(right - left) // 800))
    return [f[top:bottom:step, left:right:step] for f in frames]


def record(env, actions_or_model, path, obs, markers=None):
    """Step env with a list of actions or a model, write the rendered frames to a GIF."""
    import imageio
    from PIL import Image, ImageDraw

    if markers:
        add_markers(env.unwrapped.root, markers)
    frames = []  # not the reset: the scenes are drawn from the first step
    done, step = False, 0
    while not done:
        if isinstance(actions_or_model, list):
            if step == len(actions_or_model):
                break
            action = actions_or_model[step]
        else:
            action, _ = actions_or_model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(frame(env))
        done, step = terminated or truncated, step + 1
    images = []
    for image, (_, label) in zip(crop([f for f, _ in frames]), frames):
        picture = Image.fromarray(np.ascontiguousarray(image))
        if label:
            ImageDraw.Draw(picture).text((10, 10), label, fill=(255, 255, 255), font_size=24)
        images.append(np.asarray(picture))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, duration=100, loop=0)
    print(path, flush=True)


def isolated(function, *args):
    """Run function in a process of its own: SOFA's draw tool keeps the sphere buffers of the first GL context of a
    process (the next viewers draw garbage), and the scenes import their modules by bare name (Controller, ...)."""
    process = multiprocessing.get_context("spawn").Process(target=exit_after, args=(function, *args))
    process.start()
    process.join()


def exit_after(function, *args):
    function(*args)
    os._exit(0)  # skip SOFA's teardown


def outdated(video, source):
    return not video.exists() or video.stat().st_mtime < source.stat().st_mtime


def benchmark_video(algo, env_id, run, video):
    import gymnasium
    from stable_baselines3 import PPO, SAC

    import sofagym.envs  # noqa: F401, registers the environments

    env = gymnasium.make(env_id, render_mode="rgb_array")
    model = {"PPO": PPO, "SAC": SAC}[algo].load(run / "best", device="cpu")
    obs, _ = env.reset(seed=EPISODE_SEED)
    record(env, model, video, obs)


def benchmark_videos(results, videos):
    best = {}
    for run in sorted((results / "benchmark").glob("*_*_*")):
        if not (run / "DONE").exists() or ".old-" in run.name:
            continue
        algo, rest = run.name.split("_", 1)
        env_id = rest.rsplit("_", 1)[0]
        last = json.loads((run / "log.json").read_text())["rewards"][-1]
        if (algo, env_id) not in best or last > best[(algo, env_id)][1]:
            best[(algo, env_id)] = (run, last)
    for (algo, env_id), (run, _) in best.items():
        video = videos / f"{algo}_{env_id}.gif"
        if outdated(video, run / "best.zip"):
            isolated(benchmark_video, algo, env_id, run, video)


def planning_video(plan, video):
    result = json.loads(plan.read_text())
    env = make(result["env_id"], result["target"], planning=False, render_mode="rgb_array")
    obs, _ = env.reset(seed=result["target"])
    record(env, result["actions"], video, obs, MARKERS[result["env_id"]])


def planning_videos(results, videos):
    for plan in sorted((results / "planning").glob("*.json")):
        video = videos / f"OPD_{plan.stem}.gif"
        if outdated(video, plan):
            isolated(planning_video, plan, video)


if __name__ == "__main__":
    if not os.environ.get("DISPLAY"):
        print("No display: no videos", flush=True)
        os._exit(0)
    results = pathlib.Path(sys.argv[1])
    benchmark_videos(results, results / "videos")
    planning_videos(results, results / "videos")
