# -*- coding: utf-8 -*-
"""Record the behaviour of a SofaGym env driven by step(), to compare SOFA versions.

Runs on SOFA v23.06 (Python 3.10, SofaGym f5ad4e4) and SOFA 26, so it only uses APIs common to both.

Episodes, each starting from a reset with the same seeds:
- one per scripted action: each discrete action, or the min and max of a continuous action space, repeated
  `episode_steps` times;
- one of `random_steps` actions drawn from np.random.RandomState(0), the same in every numpy version.
Each episode records the actions, observations, rewards and dones, and the fingerprint of every MechanicalObject
(see recorder.py) after the reset and after each step. Steps are simulation steps (time / dt).

    python tests/regression/env_recorder.py <env_id> <out.json> <episode_steps> <random_steps> [<video_dir>]

With a video directory (needs a display), each episode is also rendered to <video_dir>/<episode>.gif.
After each reset: RECORD_NO_CONTACT=1 deactivates the collision models, RECORD_GRAVITY ("x,y,z") replaces the gravity
and RECORD_CLAMP_CABLES=1 keeps the cables from pushing, as the SOFA 26 update does (to record f5ad4e4).
RECORD_EPISODES (json, {name: [actions]}) replaces the scripted episodes, and RECORD_RANDOM_HOLD repeats each random
action this many times.
"""

import importlib
import json
import os
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recorder  # noqa: E402


def scripted_actions(space, steps):
    if hasattr(space, "n"):
        return [(f"action {a}", [a] * steps) for a in range(space.n)]
    return [
        (name, [np.asarray(bound, dtype=float).tolist()] * steps)
        for name, bound in (("min", space.low), ("max", space.high))
    ]


def random_actions(space, steps, hold=1):
    """`steps` random actions, each one repeated `hold` times."""
    rng = np.random.RandomState(0)
    if hasattr(space, "n"):
        actions = [int(rng.randint(space.n)) for _ in range(-(-steps // hold))]
    else:
        actions = [rng.uniform(space.low, space.high).tolist() for _ in range(-(-steps // hold))]
    return [action for action in actions for _ in range(hold)][:steps]


def record_scene(root, episode):
    recorder.sample(root, episode["objects"], int(round(root.time.value / root.dt.value)))


def render(env, frames):
    if frames is not None:
        # Every 4th pixel keeps the GIFs small.
        frames.append(np.asarray(env.render("rgb_array"))[::4, ::4, :3])


def disable_contacts(root):
    nodes = [root]
    while nodes:
        node = nodes.pop()
        nodes.extend(node.children)
        for obj in node.objects:
            if obj.getClassName().endswith("CollisionModel"):
                obj.active.value = False


def clamp_cables(root):
    """Add a controller keeping each cable above min(its initial value, 0), after the commands of each step."""
    import Sofa.Core

    cables, nodes = [], [root]
    while nodes:
        node = nodes.pop()
        nodes.extend(node.children)
        cables += [obj for obj in node.objects if obj.getClassName() == "CableConstraint"]
    floors = [min(float(cable.value.value[0]), 0.0) for cable in cables]

    class CableClamp(Sofa.Core.Controller):
        def onAnimateBeginEvent(self, event):
            for cable, floor in zip(cables, floors):
                if cable.value.value[0] < floor:
                    cable.value = [floor]

    root.addObject(CableClamp(name="CableClamp"))


def run_episode(env, actions, video=None):
    # f5ad4e4 did not reset splib3's animation manager (the module is shadowed by its animate function): from the
    # second scene, startup animations (e.g. the Maze tripod) went to the stale manager of the previous scene.
    importlib.import_module("splib3.animation.animate").manager = None
    env.seed(0)
    random.seed(0)
    np.random.seed(0)
    # Since the SOFA 26 update, the env seeds the global generators of the scene at each reset (from its own seed): seed
    # them as when the references were recorded, at 0 before the reset.
    if hasattr(env, "draw_scene_seed"):
        inner = getattr(env, "env", env)
        inner.draw_scene_seed = lambda: inner.config.update(scene_seed=0)
    episode = {
        "requested_actions": len(actions),
        "actions": [],
        # gymnasium's reset returns (obs, info), gym's obs.
        "observations": [
            np.asarray(reset[0] if isinstance(reset := env.reset(), tuple) else reset, dtype=float).ravel().tolist()
        ],
        "rewards": [],
        "dones": [],
        "objects": {},
    }
    if os.environ.get("RECORD_NO_CONTACT"):
        disable_contacts(env.root)
    if os.environ.get("RECORD_GRAVITY"):
        env.root.gravity.value = [float(g) for g in os.environ["RECORD_GRAVITY"].split(",")]
    if os.environ.get("RECORD_CLAMP_CABLES"):
        clamp_cables(env.root)
    record_scene(env.root, episode)
    # The viewer draws nothing before the first step (black frame), the video starts after it.
    frames = [] if video else None
    for action in actions:
        # gymnasium's step returns (obs, reward, terminated, truncated, info), gym's (obs, reward, done, info).
        observation, reward, done = env.step(np.asarray(action) if isinstance(action, list) else action)[:3]
        episode["actions"].append(action)
        episode["observations"].append(np.asarray(observation, dtype=float).ravel().tolist())
        episode["rewards"].append(float(reward))
        episode["dones"].append(bool(done))
        record_scene(env.root, episode)
        render(env, frames)
        if done:
            break
    if video:
        import imageio

        pathlib.Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video, frames, duration=100, loop=0)
    return episode


def env_class(env_id):
    """The class of a registered env: with gymnasium since the SOFA 26 update, with gym before (the references)."""
    import sofagym.envs  # noqa: F401, registers the environments

    try:
        import gymnasium
        from gymnasium.envs.registration import load_env_creator

        if env_id in gymnasium.registry:
            return load_env_creator(gymnasium.registry[env_id].entry_point)
    except ImportError:
        pass
    import gym
    from gym.envs.registration import load

    return load(gym.envs.registry[env_id].entry_point)


def record(env_id, episode_steps, random_steps, video_dir=None):
    random.seed(0)
    np.random.seed(0)
    # The env only keeps what the viewer draws with render=1.
    env = env_class(env_id)({"render": 1} if video_dir else None)
    scripted = (
        list(json.loads(os.environ["RECORD_EPISODES"]).items())
        if os.environ.get("RECORD_EPISODES")
        else scripted_actions(env.action_space, episode_steps)
    )
    hold = int(os.environ.get("RECORD_RANDOM_HOLD") or 1)
    episodes = scripted + [("random", random_actions(env.action_space, random_steps, hold))]
    return {
        "env": env_id,
        "episodes": {
            name: run_episode(env, actions, video_dir and os.path.join(video_dir, name.replace(" ", "_") + ".gif"))
            for name, actions in episodes
        },
    }


if __name__ == "__main__":
    env_id, out, episode_steps, random_steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    video_dir = sys.argv[5] if len(sys.argv) > 5 else None
    pathlib.Path(out).write_text(json.dumps(record(env_id, episode_steps, random_steps, video_dir)))
    sys.stdout.flush()
    # Skip SOFA's teardown, which crashes or hangs for some scenes of f5ad4e4.
    os._exit(0)
