# -*- coding: utf-8 -*-
"""runSofa level of the behaviour regression tests: record scenes with recorder.py and compare recordings.

Record the references (SOFA v23.06 build, SofaGym f5ad4e4 checkout):
    python tests/regression/runsofa.py --sofa-root ~/workspace/sofa2306/build \
        --sofagym ~/workspace/SG_main/SofaGym --out tests/regression_reference/runsofa
"""

import argparse
import json
import math
import os
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).parent
REFERENCES = HERE.parent / "regression_reference" / "runsofa"

# Steps per scene, chosen from 1000-step recordings on v23.06 (see README.md): the step where 99% of the motion is
# done, rounded up, except chaotic scenes (stopped before v23.06 and v26 diverge), static scenes and TrunkCup (cost).
STEPS = {
    "BubbleMotion/BubbleMotionScene": 100,
    "CartPole/CartPoleScene": 100,
    "CartStem/CartStemScene": 60,
    "CartStemContact/CartStemContactScene": 20,
    "CatchTheObject/CatchTheObjectScene": 100,
    "CatheterBeam/CatheterBeamScene": 20,
    "CTR/CTRScene": 100,
    "Diamond/DiamondScene": 50,
    "Gripper/GripperScene": 150,
    "Maze/MazeScene": 100,
    "SimpleMaze/SimpleMazeScene": 50,
    "StemPendulum/StemPendulumScene": 200,
    "Trunk/TrunkScene": 400,
    "TrunkCup/TrunkCupScene": 200,
}
EVERY = 10
# Allowed difference, as a fraction of the scene size (diagonal of the initial bounding box of all 3D objects).
TOLERANCE = 0.02
# Allowed rotation between the reference and new orientations of the end points of rigid objects, in radians.
ANGLE_TOLERANCE = 0.1
SCENE_TOLERANCE = {
    # The trunk stays within 1%, the cup it grasps by friction contact peaks at 2.7% during the contact.
    "TrunkCup/TrunkCupScene": 0.04,
}
# End points ("first" or "last") of rigid objects whose orientation is only compared from a given step, per scene.
ORIENTATION_FROM_STEP = {
    # The base orientation (ry=-90 in f5ad4e4) was overwritten by the controller on v23.06: the base stayed at
    # identity and the undeployed tubes rotated from identity to their rest shape in ~100 steps. It is now the
    # startingPos of the controller. The tips match from step 100 on, and a pushed tube deploys identically.
    "CTR/CTRScene": {("/InstrumentCombined/DOFs", "first"): math.inf, ("/InstrumentCombined/DOFs", "last"): 100},
}
# Gravity fixed by the SOFA 26 update (was 9.81 or 9180 in mm scenes, off for CTR): the f5ad4e4 scenes are recorded
# with it.
GRAVITY = {
    "CTR/CTRScene": [0.0, 0.0, 0.0],
    "Diamond/DiamondScene": [0.0, 0.0, -9810.0],
    "Gripper/GripperScene": [0.0, -9810.0, 0.0],
    "TrunkCup/TrunkCupScene": [0.0, -9810.0, 0.0],
}
# Objects whose number of points changed on purpose: only their end points (base and tip) are compared.
RESAMPLED = {
    "/InstrumentCombined/DOFs": "the beam grid of the instruments has as many nodes as the controller needs since the "
    "SOFA 26 update (106 for CTR, 146 for CatheterBeam, was 60)",
}
# Objects whose expected change prevents the comparison.
IGNORED = {
    "/InstrumentCombined/Collis/CollisionDOFs": "BeamAdapter #203 sizes the collision model of MultiAdaptiveBeamMapping "
    "for all the instruments instead of the most discretized one",
    "/Cube/mstate": "the cube of the Gripper is a solid of hexahedra since the SOFA 26 update (a hollow surface before)",
}


def record(scene, out, sofa_root, sofagym_root, timeout=600):
    """Run `scene` (e.g. "CTR/CTRScene") in runSofa batch mode and return the recording, or raise with the log."""
    sofa_root, sofagym_root, out = (pathlib.Path(p).resolve() for p in (sofa_root, sofagym_root, out))
    env = dict(
        os.environ,
        SOFA_ROOT=str(sofa_root),
        PYTHONPATH=os.pathsep.join([str(sofagym_root), str(sofa_root / "lib" / "python3" / "site-packages")]),
        RECORD_SCENE=str(sofagym_root / "sofagym" / "envs" / f"{scene}.py"),
        RECORD_OUT=str(out),
        RECORD_STEPS=str(STEPS[scene]),
        RECORD_EVERY=str(EVERY),
        RECORD_GRAVITY=",".join(map(str, GRAVITY.get(scene, []))),
    )
    out.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [str(sofa_root / "bin" / "runSofa"), "-l", "SofaPython3", "-g", "batch", str(HERE / "recorder.py")],
            cwd=sofagym_root,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        log = result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:
        log = f"Timeout after {timeout}s\n{error.stdout or ''}{error.stderr or ''}"
    # runSofa exits with 0 even when the scene fails to load, the recorder only writes `out` after all the steps.
    if not out.exists():
        raise RuntimeError(f"runSofa did not record {scene}:\n{log[-5000:]}")
    return json.loads(out.read_text())


def scene_scale(recording):
    points = [s for o in recording["objects"].values() if len(o["samples"][0]["min"]) >= 3 for s in o["samples"][:1]]
    low = np.min([p["min"][:3] for p in points], axis=0)
    high = np.max([p["max"][:3] for p in points], axis=0)
    return float(np.linalg.norm(high - low))


def compare(reference, recording, scene):
    """Return the differences larger than `tolerance`, as readable strings (empty when the behaviour matches).

    3D (and rigid) objects are compared on their first 3 coordinates, relative to the scene size: centroid, bounding
    box and end points at every sample. Other objects (1D, 2D actuators) are compared relative to their own range.
    The orientations of the end points of rigid objects are compared too: undeployed instruments, for instance, only
    differ by their orientation.
    """
    tolerance = SCENE_TOLERANCE.get(scene, TOLERANCE)
    orientation_from_step = ORIENTATION_FROM_STEP.get(scene, {})
    scale = scene_scale(reference)
    failures = []
    for path, ref in reference["objects"].items():
        if path in IGNORED:
            continue
        new = recording["objects"].get(path)
        if new is None:
            failures.append(f"{path}: missing")
            continue
        if new["size"] != ref["size"] and path not in RESAMPLED:
            failures.append(f"{path}: {new['size']} points instead of {ref['size']}")
            continue
        keys = ("first", "last") if path in RESAMPLED else ("centroid", "min", "max", "first", "last")
        geometric = len(ref["samples"][0]["min"]) >= 3
        coords = slice(0, 3) if geometric else slice(None)
        if geometric:
            object_scale = scale
        else:
            values = np.array([s[k] for s in ref["samples"] for k in ("min", "max")])
            object_scale = max(float(np.ptp(values)), float(np.abs(values).max()), 1e-9)
        worst = (0.0, None, None)
        for ref_sample, new_sample in zip(ref["samples"], new["samples"]):
            for key in keys:
                diff = np.subtract(new_sample[key][coords], ref_sample[key][coords]) / object_scale
                error = float(np.abs(diff).max())
                if error > worst[0]:
                    worst = (error, ref_sample["step"], key, ref_sample[key][coords], new_sample[key][coords])
        if worst[0] > tolerance:
            error, step, key, expected, got = worst
            failures.append(
                f"{path}: {key} at step {step} differs by {error:.1%} of the scale ({object_scale:.4g}): "
                f"expected {np.round(expected, 4).tolist()}, got {np.round(got, 4).tolist()}"
            )
        if ref["template"].startswith("Rigid3"):
            angles = [
                (2 * np.arccos(min(1.0, abs(float(np.dot(r[key][3:7], n[key][3:7]))))), r["step"], key)
                for r, n in zip(ref["samples"], new["samples"])
                for key in ("first", "last")
                if r["step"] >= orientation_from_step.get((path, key), 0)
            ]
            angle, step, key = max(angles, default=(0.0, None, None))
            if angle > ANGLE_TOLERANCE:
                failures.append(f"{path}: {key} at step {step} is rotated by {np.degrees(angle):.1f} degrees")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sofa-root", required=True, help="SOFA build or install directory")
    parser.add_argument("--sofagym", required=True, help="SofaGym checkout whose scenes are recorded")
    parser.add_argument("--out", default=str(REFERENCES), help="Output directory")
    parser.add_argument("scenes", nargs="*", default=sorted(STEPS), help="Scenes to record, default all")
    args = parser.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        out = pathlib.Path(args.out) / f"{pathlib.Path(scene).name}.json"
        record(scene, out, args.sofa_root, args.sofagym)
        print(f"{scene}: {out}")
