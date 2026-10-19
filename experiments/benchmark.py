# -*- coding: utf-8 -*-
"""RL benchmark of the SofaGym paper (Schegg, Ménager et al., Soft Robotics 2023): one algorithm, env and seed.

Follows the protocol of the original training script (rl.py of SofaGym, 2022): epochs of timer_limit * 20 env steps,
each followed by 5 test episodes (stochastic policy); the mean total and final rewards of the tests are logged after each
epoch. The model (and SAC's replay buffer) is saved after each epoch, so that a run resumes from its last epoch.

Usage:
-----
    python experiments/benchmark.py cartstem-v0 PPO 0 --out experiments/results
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Hyperparameters of the paper (Appendix C) and of the original rl.py (epochs). The scale factors of the paper (10, 30,
# 10) are those of the env configs. env id: (timer_limit, epochs, batch size).
ENVS = {
    "cartstemcontact-v0": (30, 600, 200),
    "cartstem-v0": (80, 200, 256),
    "catchtheobject-v0": (30, 200, 256),
}
GAMMA = 0.99
LEARNING_RATE = 1e-4
LAYERS = [512, 512, 512]
N_TESTS = 5
PPO_ENVS = 4  # SubprocVecEnv of the original PPO runs
# Small networks, as the paper (CPU): a GPU would only add a CUDA context (~1 GB of RAM) to each of the parallel runs.
DEVICE = "cpu"


def test(env, model, n_tests=N_TESTS):
    """Mean total reward and mean final reward over n_tests episodes, as in the original rl.py."""
    totals, finals = [], []
    for _ in range(n_tests):
        obs, _ = env.reset()
        rewards, done = [], False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated
        totals.append(sum(rewards))
        finals.append(rewards[-1])
    return float(np.mean(totals)), float(np.mean(finals))


def write_json(path, data):
    # Written then renamed: a crash never leaves a truncated log.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def run(env_id, algo, seed, out, max_epochs):
    """Train at most max_epochs epochs (the process then exits, see --epochs), return whether the run is finished."""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from agents.utils import make_env

    timer_limit, epochs, batch_size = ENVS[env_id]
    epoch_steps = timer_limit * 20
    run_dir = pathlib.Path(out) / f"{algo}_{env_id}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path, latest = run_dir / "log.json", run_dir / "latest"
    log = (
        json.loads(log_path.read_text())
        if log_path.exists()
        else {"rewards": [], "final_rewards": [], "steps": [], "timesteps": []}
    )

    # Seeds as in the original: seed * 10 for the test env (and SAC's env), + rank for the PPO envs.
    test_env = make_env(env_id, 0, seed * 10)()
    if algo == "SAC":
        env = make_env(env_id, 0, seed * 10)()
        kwargs = dict(ent_coef="auto", learning_starts=500, policy_kwargs=dict(net_arch=dict(pi=LAYERS, qf=LAYERS)))
        Algo = SAC
    else:
        env = SubprocVecEnv([make_env(env_id, rank, seed) for rank in range(PPO_ENVS)])
        kwargs = dict(n_steps=epoch_steps, policy_kwargs=dict(net_arch=dict(pi=LAYERS, vf=LAYERS)))
        Algo = PPO

    if log["rewards"] and latest.with_suffix(".zip").exists():
        model = Algo.load(latest, env=env, device=DEVICE)
        if algo == "SAC":
            model.load_replay_buffer(run_dir / "latest_replay_buffer")
        print(f"Resume {run_dir.name} at epoch {len(log['rewards'])}/{epochs}", flush=True)
    else:
        log = {"rewards": [], "final_rewards": [], "steps": [], "timesteps": []}
        model = Algo(
            "MlpPolicy",
            env,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            batch_size=batch_size,
            seed=seed * 10,
            device=DEVICE,
            **kwargs,
        )

    first = len(log["rewards"])
    for epoch in range(first, min(epochs, first + max_epochs)):
        model.learn(total_timesteps=epoch_steps, reset_num_timesteps=False, log_interval=20)
        reward, final_reward = test(test_env, model)
        if not log["rewards"] or reward >= max(log["rewards"]):
            model.save(run_dir / "best")
        model.save(latest)
        if algo == "SAC":
            model.save_replay_buffer(run_dir / "latest_replay_buffer")
        log["rewards"].append(reward)
        log["final_rewards"].append(final_reward)
        # As the original, steps counts epochs of timer_limit * 20 steps (the PPO envs together do 4 times more).
        log["steps"].append(epoch_steps * (epoch + 1))
        log.setdefault("timesteps", []).append(int(model.num_timesteps))
        write_json(log_path, log)
        print(f"{run_dir.name} epoch {epoch + 1}/{epochs}: reward {reward:.3f}, final {final_reward:.3f}", flush=True)

    env.close()
    test_env.close()
    return len(log["rewards"]) >= epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env_id", choices=sorted(ENVS))
    parser.add_argument("algo", choices=["SAC", "PPO"])
    parser.add_argument("seed", type=int)
    parser.add_argument("--out", default="experiments/results/benchmark")
    # Each new SOFA scene (episode) leaks a little memory (~0.25 MB): the process exits regularly, then resumes.
    parser.add_argument("--epochs", type=int, default=50, help="Epochs to train before exiting (to resume)")
    args = parser.parse_args()
    finished = run(args.env_id, args.algo, args.seed, args.out, args.epochs)
    print("BENCHMARK_DONE" if finished else "BENCHMARK_CONTINUE", flush=True)
    # Skip SOFA's teardown.
    os._exit(0)
