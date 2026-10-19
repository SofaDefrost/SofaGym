# Reproducing the experiments of the SofaGym paper

Ménager, Schegg et al., *SofaGym: An open platform for Reinforcement Learning based on Soft Robot simulations*, Soft
Robotics 2023 (hal-03778189).

```bash
experiments/reproduce.sh                           # everything, resumable
SEEDS="0" ENVS="cartstem-v0" experiments/reproduce.sh  # a subset
```

To run it in the background, independently of the terminal, and with a memory cap so that a problem kills the runs
rather than the desktop (they resume from their checkpoints). `CPUS` fits the cap (~2.3 GB per process): the default
only looks at the memory of the machine.

```bash
systemd-run --user --unit=sofagym-reproduce -p MemoryMax=22G -p MemorySwapMax=0 --working-directory=$PWD \
    --setenv=PATH=$(dirname $(which python)):/usr/bin:/bin --setenv=DISPLAY=$DISPLAY --setenv=XAUTHORITY=$XAUTHORITY \
    --setenv=CPUS=9 bash -c 'set -a; . ./.env; set +a; experiments/reproduce.sh >> experiments/results/reproduce.log 2>&1'
systemctl --user status sofagym-reproduce   # follow it; stop it with systemctl --user stop sofagym-reproduce
```

Results go to `experiments/results/` (not versioned): one directory per run (`<algo>_<env>_<seed>`) with its model
(`latest`, `best`), its test rewards (`log.json`) and its output, the figures (`<env>_rewards.png`,
`<env>_final_rewards.png`), and with a display the videos (`videos/`, `video.py`): an episode of the best agent of each
algorithm and env, and the plan of OPD for each target. Rerunning the script resumes the unfinished runs and skips the finished ones; a run whose
code changed (its env, SofaGym's core, the benchmark) restarts from scratch.

## RL baseline (section 5.1, figure 2)

`benchmark.py` follows the paper (Appendix C) and the original training script of SofaGym (2022):

- algorithms: PPO and SAC of Stable Baselines3 (SAC only with continuous actions, so not on CartStem);
- gamma 0.99, learning rate 1e-4, 3 layers of 512 (policy and value), batch size 256 (CartStem, CatchTheObject) or
  200 (CartStemContact), SAC with automatic entropy and 500 learning starts, PPO on 4 envs with `n_steps` of an epoch;
- epochs of `timer_limit * 20` steps (CartStem 1600, CartStemContact and CatchTheObject 600), each followed by 5 tests
  with the stochastic policy; 200 epochs (CartStem, CatchTheObject) and 600 (CartStemContact);
- 6 seeds.

The steps of the figures count epochs of `timer_limit * 20` steps, as in the paper; the PPO runs do 4 times more env
steps (4 envs, `timesteps` in `log.json`).

Reference results (figure 2, total reward of the tests at the end of the training):

| Env | PPO | SAC |
|---|---|---|
| CartStem (300k steps) | ~70-75 (max 80) | - |
| CartStemContact (~200k steps) | oscillates, below SAC | ~-5, best |
| CatchTheObject (120k steps) | ~-3 | ~-3 |

Differences with the paper:

- Only the vanilla PPO and SAC: the paper's AVEC variants (PPO-AVEC, SAC-AVEC) are dropped. No maintained library implements
  AVEC, and SB3's SAC has no state-value network to apply SAC-AVEC to.
- SAC is tested on its own env, not on its training env (the original script tested on the training env).
- SOFA 26 and the fixes of the scenes since the paper: CartStem's stem bends more than on SOFA v23.06 (see
  `tests/regression/README.md`), which may change its results.

## Planning (sections 5.3 and 5.4)

`planning.py` runs OPD (rl-agents' DeterministicPlannerAgent) with the server in planning mode on one target, then
replays the actions locally to measure the final distance. Targets: 10 random ones for Diamond and ConcentricTubeRobot,
the 2 goals of Trunk, the 4 corners of SimpleMaze. The paper gives neither the budget nor the targets: the budget is 4
expansions of all the actions per real step, gamma 0.9. `summarize_planning.py` writes `planning/summary.md`.

ConcentricTubeRobot has 16 expansions (192): with 4 (48), 3 targets out of 10 were reached. Its tubes are damped
(rayleighStiffness 0.2) and a step lasts 0.1 s: the tip follows a command over ~4 steps, so a step is rewarded for
the previous actions. Its reward (improvement over the best distance so far, floored at 0) is 0 once the tip stalls,
and OPD looking ~2 actions ahead then picks at random: 6 of the 7 failed targets came within 3.7-5.7 mm of the goal
(reached at 3 mm) and drifted away. Same reward, damping and targets as f5ad4e4.

Reference results: Diamond, OPD 3.13 mm of mean error in 38.24 s per trajectory (QP: 13.24 mm, 3.93 s, table 1);
ConcentricTubeRobot, 100% of the targets reached; SimpleMaze, the 4 corners reached; Trunk, similar to Diamond.

Not run yet: the QP side of table 1 (`DiamondSceneInverse`).

## Not reproduced yet

- The multigait (section 5.5): its reduced model needs MechanicalMatrixMapperMOR, removed from SOFA.
- The gripper (section 5.2, PPO with an LSTM, random pulling forces, pre-training by behavior cloning): only the LSTM
  size (128) is given; the gripper env changed since (base translation, solid cube, cable speed).
