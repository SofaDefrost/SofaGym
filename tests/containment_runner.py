# -*- coding: utf-8 -*-
"""Push each instrument of an env (CTR, CatheterBeam) and check that it stays inside the cavity or vessels.

Inside means a generalized winding number (Jacobson et al. 2013) of at least 0.5 with respect to the collision mesh,
its openings capped. Walking along the instrument from its base, it may only leave through a cap (an opening of the
mesh, e.g. the far end of the CTR cavity): the points beyond are not checked. Leaving elsewhere means it went through
a wall. The points closer than ENTRY mm to its base are not checked: the instruments enter through an opening.

With --video, each push is also rendered at full resolution to <video>/instrument_<i>.gif, cropped to the pixels that
change (the instruments are thin).

Usage:
-----
    python tests/containment_runner.py catheter_beam-v0 --insertion 450
"""

import argparse
import importlib
import os
import pathlib
import random

import numpy as np

# Insertion of each instrument (instrument i is pushed by action 4 * i).
PUSH_ACTIONS = (0, 4, 8)
ENTRY = 10.0
MAX_STEPS = 200


def winding_numbers(points, vertices, triangles):
    """Generalized winding number of each point: ~1 inside the mesh, ~0 outside, even with a few openings."""
    a, b, c = (vertices[triangles[:, i]][None] - points[:, None] for i in range(3))
    la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
    det = np.einsum("pti,pti->pt", a, np.cross(b, c))
    div = (
        la * lb * lc
        + np.einsum("pti,pti->pt", a, b) * lc
        + np.einsum("pti,pti->pt", b, c) * la
        + np.einsum("pti,pti->pt", c, a) * lb
    )
    return np.arctan2(det, div).sum(axis=1) / (2 * np.pi)


def cap_openings(vertices, triangles):
    """Vertices and triangles closing each opening of the mesh with a fan to its center, and the cap triangles."""
    edges = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, index, count = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    # Border edges, as oriented in their triangle: the caps take them reversed, to keep the orientation of the mesh.
    border = edges[count[index.ravel()] == 1]
    # Openings: connected components of the border edges.
    label = np.arange(len(vertices))
    changed = True
    while changed:
        low = np.minimum(label[border[:, 0]], label[border[:, 1]])
        changed = bool((label[border] != low[:, None]).any())
        np.minimum.at(label, border[:, 0], low)
        np.minimum.at(label, border[:, 1], low)
    caps, centers = [], []
    for k, opening in enumerate(np.unique(label[border[:, 0]])):
        loop = border[label[border[:, 0]] == opening]
        caps += [[b, a, len(vertices) + k] for a, b in loop]
        centers.append(vertices[np.unique(loop)].mean(axis=0))
    return np.vstack([vertices, centers]), np.vstack([triangles, caps]), np.array(caps)


def crosses(p, q, vertices, triangles):
    """Whether the segment pq crosses one of the triangles (Moller-Trumbore)."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    d, e1, e2 = q - p, b - a, c - a
    h = np.cross(d, e2)
    det = np.einsum("ti,ti->t", e1, h)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    s = p - a
    u = inv * np.einsum("ti,ti->t", s, h)
    w = np.cross(s, e1)
    v = inv * (w @ d)
    t = inv * np.einsum("ti,ti->t", e2, w)
    return bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)).any())


def collision_mesh(root):
    """Vertices (transforms applied) and triangles of the cavity (CTR) or vessels (CatheterBeam)."""
    node = root.Vessels if hasattr(root, "Vessels") else root.CollisionModel
    state = node.getMechanicalState()
    vertices = (state.position if state is not None else node.meshLoader.position).value
    return np.array(vertices)[:, :3], np.array(node.meshLoader.triangles.value)


def save_cropped(frames, path, margin=60):
    """Write the frames as a GIF cropped to the pixels that change between them, plus a margin, enlarged if small."""
    import imageio

    # The first frame is rendered before the visual models are updated.
    stack = np.array(frames[1:])
    ys, xs = np.where(np.abs(stack.astype(int) - stack[0]).max(axis=(0, 3)) > 30)
    if len(ys):
        stack = stack[:, max(ys.min() - margin, 0) : ys.max() + margin, max(xs.min() - margin, 0) : xs.max() + margin]
    zoom = max(1, 1000 // max(stack.shape[1:3]))
    stack = stack.repeat(zoom, axis=1).repeat(zoom, axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(stack), duration=150, loop=0)


def run(env_id, insertion=None, steps=None, video=None):
    import gym
    from gym.envs.registration import load

    import sofagym.envs  # noqa: F401, registers the environments

    importlib.import_module("splib3.animation.animate").manager = None
    random.seed(0)
    np.random.seed(0)
    env = load(gym.envs.registry[env_id].entry_point)({"render": 1} if video else None)
    failures = []
    for action in PUSH_ACTIONS:
        env.seed(0)
        env.reset()
        vertices, triangles, caps = cap_openings(*collision_mesh(env.root))
        frames = [np.asarray(env.render("rgb_array"))[..., :3]] if video else None
        instrument, step = action // 4, 0
        tip = env.root.InstrumentCombined.m_ircontroller.xtip
        while (step < steps) if steps else (tip.value[instrument] < insertion):
            env.step(action)
            step += 1
            if video:
                frames.append(np.asarray(env.render("rgb_array"))[..., :3])
            # The collision points, from the base to the tip (the unused ones are stacked at the base).
            points = np.array(env.root.InstrumentCombined.Collis.CollisionDOFs.position.value)[:, :3]
            points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
            base = np.array(env.root.InstrumentCombined.DOFs.position.value)[0, :3]
            if np.linalg.norm(points[0] - base) > np.linalg.norm(points[-1] - base):
                points = points[::-1]
            far = np.linalg.norm(points - base, axis=1) > ENTRY
            if not far.any():
                continue
            points = points[np.argmax(far) :]
            inside = np.abs(winding_numbers(points, vertices, triangles)) >= 0.5
            wall = next(
                (
                    k
                    for k in range(len(points))
                    if not inside[k] and not (k and inside[k - 1] and crosses(points[k - 1], points[k], vertices, caps))
                ),
                None,
            )
            exits = next((k for k in range(1, len(points)) if inside[k - 1] and not inside[k]), None)
            if wall is not None and (exits is None or wall <= exits):
                failures.append(
                    f"instrument {instrument}: through a wall at {tip.value[instrument]:.0f} mm, "
                    f"near {np.round(points[wall], 1).tolist()}"
                )
                break
            if step == MAX_STEPS:
                failures.append(f"instrument {instrument}: stuck at {tip.value[instrument]:.0f} mm")
                break
        print(
            f"instrument {instrument}: inserted {tip.value[instrument]:.0f} mm in {step} steps"
            + (", out through an opening" if exits is not None else ""),
            flush=True,
        )
        if video:
            save_cropped(frames, pathlib.Path(video) / f"instrument_{instrument}.gif")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="concentrictuberobot-v0 or catheter_beam-v0")
    length = parser.add_mutually_exclusive_group(required=True)
    length.add_argument("--insertion", type=float, help="Length to insert each instrument (mm)")
    length.add_argument("--steps", type=int, help="Env steps pushing each instrument")
    parser.add_argument("--video", help="Directory of the GIFs (needs a display)")
    args = parser.parse_args()
    failures = run(args.env_id, args.insertion, args.steps, args.video)
    print("\n".join(failures) if failures else "CONTAINED_OK", flush=True)
    # Skip SOFA's teardown.
    os._exit(0)
