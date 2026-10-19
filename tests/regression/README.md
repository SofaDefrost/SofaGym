# Behaviour regression tests

The scenes of SofaGym `f5ad4e4` (last commit before the SOFA 26 update), run on SOFA v23.06, are the references.
The current scenes, run on SOFA 26, must behave like them within tolerances: results are not expected to be
identical, but errors such as an instrument deploying along the wrong axis must be caught.

## runSofa level

`recorder.py` is a runSofa scene wrapping the `createScene` of a SofaGym scene with its default config. Every 10
steps it samples each MechanicalObject: centroid, bounding box, first and last points (e.g. the tip of a wire). It
seeds `random` and `np.random` (some scenes draw their initial state), skips the transient contact objects, writes
the recording after the requested steps and exits without SOFA's teardown (the f5ad4e4 scenes crash or hang there).

`test_regression_runsofa.py` records the current scenes and compares them to `tests/regression_reference/runsofa/*.json`:

- same objects, with the same number of points;
- centroid, bounding box and end points within 2% of the scene size (diagonal of its initial bounding box) at
  every sample; 1D/2D objects within 2% of their own range;
- orientations of the end points of rigid objects within 0.1 rad.

Exceptions (tolerance, ignored objects, orientations compared from a given step) are in `runsofa.py`, each with
its reason.

### Number of steps per scene

Chosen from 1000-step recordings on v23.06 (motion extent = largest displacement of any object over 1000 steps;
t99 = step where every object reached 99% of its extent; cost on SOFA 26):

| Scene | Steps | Why |
|---|---|---|
| BubbleMotion | 100 | t99 = 40 |
| CartPole | 100 | v26 departs from v23.06 at step 130 (pole falls on an unstable equilibrium, chaotic) |
| CartStem | 60 | inverted pendulum: v26 departs from v23.06 by 0.1% at step 20 and 1% at step 80 |
| CartStemContact | 20 | static without actions, only checks the initial geometry |
| CatchTheObject | 100 | the ball falls past the cart after ~20 steps |
| CatheterBeam | 20 | static without actions (no gravity), only checks the initial geometry |
| CTR | 100 | t99 = 100 (the undeployed tubes settle) |
| Diamond | 50 | t99 = 10 |
| Gripper | 150 | t99 = 120 |
| Maze | 100 | t99 = 50 |
| SimpleMaze | 50 | t99 = 10 |
| StemPendulum | 200 | t99 = 30, one swing back, cheap (0.8 ms/step) |
| Trunk | 400 | t99 = 320 |
| TrunkCup | 200 | t90 = 170; t99 = 760 would cost 45 s (58 ms/step), the remaining motion is slow creep |

The whole runSofa level takes about 30 s. Scenes without actions don't exercise actuation (e.g. tube insertion):
the environment level, driving `step()` with scripted actions, covers it.

No reference: DiamondSceneInverse and MultiGaitRobot were already broken on v23.06.

### Recording the references

SOFA v23.06 with the plugins of the time is built in `~/workspace/sofa2306/build` (conda env `sofa2306`), and
`~/workspace/SG_main/SofaGym` is a checkout of `f5ad4e4`. The old code loads Cosserat as `CosseratPlugin`:
`lib/libCosseratPlugin.so` is a symlink to `libCosserat.so` in the build.

    python tests/regression/runsofa.py --sofa-root ~/workspace/sofa2306/build --sofagym ~/workspace/SG_main/SofaGym

Recording one scene: add its name, e.g. `CTR/CTRScene`. After changing the steps of a scene, re-record it.

## Environment level

`env_recorder.py` creates a registered env (gym, Python 3.10 on v23.06) and runs episodes, each from a reset with
the same seeds: one per scripted action (each discrete action, or the min and max of a continuous action space)
repeated `steps` times, then one of `random` actions drawn from `np.random.RandomState(0)`. Each episode records the
actions, observations, rewards and dones, and the scene fingerprint of the runSofa level after the reset and each
step. Scripted actions exercise the actuation (inserting a tube, pulling a cable, inflating a cavity), which scenes
without actions don't.

`test_regression_envs.py` records the current envs and compares them to `tests/regression_reference/envs/*.json`:

- same episode lengths (step where `done` is set);
- observations and rewards within 2% of their range over all the reference episodes (dimensions that barely move:
  at least 5% of the largest range of the vector);
- the scene fingerprint, as at the runSofa level.

The steps of each env are the longest prefix where v26 matched v23.06, measured over 20-step scripted episodes and
40-step random ones; the exceptions (chaotic or contact-driven behaviour, known differences) are in `envs.py`, with
their reasons.

### Videos

With a display, each episode is rendered to a GIF (viewer of the env, one frame per step), to see what the numbers
mean. Recording the references writes `tests/videos/regression/v23/<env>/<episode>.gif`; each test run writes
`tests/videos/regression/v26/...` and `tests/videos/regression/compare/<env>/<episode>.gif`, v23.06 on the left and v26 on the right, as long as the steps the test
compares. The videos are not versioned (`tests/videos/`), the references are.

### Recording the references

    python tests/regression/envs.py --sofa-root ~/workspace/sofa2306/build --sofagym ~/workspace/SG_main/SofaGym \
        --python ~/miniconda3/envs/sofa2306/bin/python

### Differences found and fixed with these tests

- SoftRobots actuator values are no longer divided by dt in SOFA 26: the pressures of BubbleMotion and
  CatchTheObject are scaled by 1/dt to behave as on v23.06.
- f5ad4e4 did not reset splib3's animation manager between scenes (the module is shadowed by its `animate`
  function), so startup animations, e.g. the Maze tripod, never ran after the first reset. Fixed in AbstractEnv
  and ServerEnv; the recorder applies the same reset to f5ad4e4.
- The viewer did not update the visual models since the SOFA 26 update: `render()` showed the first frame.
- The instruments of CTR and CatheterBeam collided as lines (proximity 0) and went through the cavity or vessel
  walls after 50-140 mm of insertion (on v23.06 too, earlier for CTR). Their collision proximity is now 0.25 (CTR,
  tubes of radius 0.15) and 0.3 (CatheterBeam, 0.36 mm wire), the smallest keeping them inside for 150 mm, checked
  by `tests/test_instrument_containment.py` (450 mm for CatheterBeam). The CTR regression case with contacts is short for this reason; the
  `-no-contact` case compares the mechanics of the tubes on whole episodes.
- The catheter of CatheterBeam had a straight tip (a 4000 mm diameter spire), it now has a curved tip as the CTR
  tubes (25 mm long, 25 mm diameter): its environment no longer has v23.06 references, the runSofa level (no
  actions) and the containment test still cover it.
- The Gripper did not move its base (translations moved the fingers, held to their rest positions), its cube was a
  hollow FEM surface the fingers squeezed flat, and its cables moved 5 mm per step. The base now moves, the cube is a
  soft solid (4x4x4 hexahedra, 200 kPa, 30 g) and the cables move 2.5 mm per step: its episodes are only recorded for
  the videos.
- The stem of CartStem had no visual model (CartStemContact draws its collision model).
- The goal of Trunk (GoalMO, drawn and in the observation) was at the goal of the reward plus the initial position
  of the GoalMO, [0, -100, 100]: it is now at the goal of the reward.

### Known differences

- CartStem: the tip of the stem follows the cart less on v26 and the stem bends over within 16 random steps, where
  it stays up for 40 on v23.06, with the same scene and Cosserat force field code. With its parameters (EI = 1.3e5
  kg.cm^3/s^2, 25 cm, 0.75 kg tip at 981 cm/s^2) the tip weight is above the Euler buckling load, so the v26
  behaviour is the expected one; the v23.06 cause is not identified.
