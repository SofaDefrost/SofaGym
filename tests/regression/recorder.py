# -*- coding: utf-8 -*-
"""Record a behavioural fingerprint of a SofaGym scene, to compare SOFA versions.

Runs on SOFA v23.06 (Python 3.10, SofaGym f5ad4e4) and SOFA 26, so it only uses APIs common to both.

As a runSofa scene, wrapping the createScene of a SofaGym scene with its default config:
    RECORD_SCENE=<scene_file> RECORD_OUT=<out.json> RECORD_STEPS=100 RECORD_EVERY=10 \
        runSofa -l SofaPython3 -g batch tests/regression/recorder.py
The parameters are environment variables because runSofa --argv differs between v23.06 and v26. RECORD_GRAVITY
("x,y,z") replaces the gravity of the scene.
runSofa exits with 0 even when the scene fails to load: a missing output file is the failure signal.
"""

import importlib.util
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import Sofa.Core


def mechanical_objects(root):
    nodes, visited = [root], set()
    while nodes:
        node = nodes.pop()
        # A node with several parents is listed under each of them.
        if node.getPathName() in visited:
            continue
        visited.add(node.getPathName())
        nodes.extend(node.children)
        for obj in node.objects:
            # Contact responses create transient MechanicalObjects with random names.
            if obj.getClassName() == "MechanicalObject" and "CollisionModel-" not in node.getPathName():
                yield obj


def summary(positions):
    """Centroid, bounding box and end points: enough to see a wrong axis, scale or offset, without the whole mesh."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    return {
        "centroid": positions.mean(axis=0).tolist(),
        "min": positions.min(axis=0).tolist(),
        "max": positions.max(axis=0).tolist(),
        "first": positions[0].tolist(),
        "last": positions[-1].tolist(),
    }


def sample(root, objects, step):
    """Append the summary of every MechanicalObject of `root` at `step` to `objects` ({path: object record})."""
    seen = {}
    for mo in mechanical_objects(root):
        positions = mo.position.value
        if len(positions) == 0:
            continue
        # Some scenes have several objects with the same path, number them in traversal order.
        path = mo.getPathName()
        seen[path] = seen.get(path, -1) + 1
        key = f"{path}#{seen[path]}" if seen[path] else path
        entry = objects.setdefault(key, {"template": mo.getTemplateName(), "size": len(positions), "samples": []})
        entry["samples"].append(dict(step=step, **summary(positions)))


class Recorder(Sofa.Core.Controller):
    """Samples every MechanicalObject each `every` steps, writes `out` after `steps` steps and exits.

    os._exit skips SOFA's teardown, which crashes or hangs for some scenes of f5ad4e4.
    """

    def __init__(self, root, out, steps, every, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root, self.out, self.steps, self.every = root, out, steps, every
        self.step = 0
        self.start = None
        self.record = {"steps": steps, "every": every, "dt": None, "objects": {}}

    def sample(self):
        sample(self.root, self.record["objects"], self.step)

    def onAnimateBeginEvent(self, event):
        if self.start is None:
            self.record["dt"] = self.root.dt.value
            self.sample()
            self.start = time.time()

    def onAnimateEndEvent(self, event):
        self.step += 1
        if self.step % self.every == 0 or self.step == self.steps:
            self.sample()
        if self.step == self.steps:
            self.record["wall_time_per_step"] = (time.time() - self.start) / self.steps
            pathlib.Path(self.out).write_text(json.dumps(self.record))
            sys.stdout.flush()
            os._exit(0)


def load_scene_module(scene_file):
    sys.path.insert(0, os.path.dirname(scene_file))
    spec = importlib.util.spec_from_file_location(pathlib.Path(scene_file).stem, scene_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def createScene(root):
    # Some scenes draw their initial state at random (StemPendulum angle, CatchTheObject cart, Gripper offset).
    random.seed(0)
    np.random.seed(0)
    load_scene_module(os.path.abspath(os.environ["RECORD_SCENE"])).createScene(root)
    if os.environ.get("RECORD_GRAVITY"):
        root.gravity.value = [float(g) for g in os.environ["RECORD_GRAVITY"].split(",")]
    root.addObject(
        Recorder(
            root,
            os.environ["RECORD_OUT"],
            int(os.environ["RECORD_STEPS"]),
            int(os.environ["RECORD_EVERY"]),
            name="BehaviourRecorder",
        )
    )
    return root
