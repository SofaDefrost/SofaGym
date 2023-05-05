# -*- coding: utf-8 -*-
"""Test the MultiGaitRobotEnv by learning a policy to move in the x direction.

Usage:
-----
    python3.7 rl_multigait.py
"""

__authors__ = "emenager, pschegg"
__contact__ = "etienne.menager@ens-rennes.fr, pierre.schegg@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020,Inria"
__date__ = "Nov 10 2020"


from stable_baselines3 import *
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Video

import gym

import sys
import os
import json
import pathlib
import numpy as np
import torch
import random
import argparse
import time
from colorama import Fore

import sofagym
from sofagym.envs import *

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


# Adapted from rl-agents
def load_environment(env_name, rank=0, seed=0, max_episode_steps=100):
    def _init():
        __import__('sofagym')
        env = gym.make(env_name)
        env.seed(seed + rank)
        env.reset()
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    return _init


def test(env, model, test, n_steps, n_episode=1, render=False):
    if render:
        env.config.update({"render":1})

    r, final_r = 0, 0

    for t in range(n_episode):
        print("Start >> Epoch", test, "- Episode", t+1)
        obs = env.reset()

        if render:
            env.render()
        
        rewards = []
        done = False
        id = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic = False)
            obs, reward, done, info = env.step(action)
            if render:
                print("Episode", t+1, "- Step ", id+1 ,"- Took action: ", action, "- Current State: ", obs, "- Got reward: ", reward, "- Done: ", done)
                env.render()
            rewards.append(reward)
            id +=1
        
        print("Done >> Episode", t+1, "- Reward = ", rewards, "- Sum reward:", sum(rewards))
        r += sum(rewards)
        final_r += reward
    
    print("[INFO]  >> Mean reward: ", r/n_test, " - Mean final reward:", final_r/n_test)

    env.close()
    
    return r/n_test, final_r/n_test


def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    return d


results_dir = "./Results_benchmark"
models_dir = f"{results_dir}/models"
logs_dir = f"{results_dir}/logs"
videos_dir = f"{logs_dir}/videos"

envs = {
        1: 'bubblemotion-v0',
        2: 'cartstem-v0',
        3: 'cartstemcontact-v0',
        4: 'catchtheobject-v0',
        5: 'concentrictuberobot-v0',
        6: 'diamondrobot-v0',
        7: 'gripper-v0',
        8: 'maze-v0',
        9: 'multigaitrobot-v0',
        10: 'simple_maze-v0',
        11: 'stempendulum-v0',
        12: 'trunk-v0',
        13: 'trunkcup-v0',
        }

algos = {
        1: 'SAC',
        2: 'PPO',
        3: 'A2C',
        4: 'DQN',
        5: 'TD3',
        6: 'DDPG'
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", help="Number of the environment",
                        type=str, required=True)
    parser.add_argument("-a", "--algo", help = "RL algorithm",
                        type=str, required=True)
    parser.add_argument("-ne", "--env_num", help = "Number of parallel envs",
                        type=int, required=False, default=4)
    parser.add_argument("-s", "--seed", help = "Seed",
                        type=int, required=False, default=0)
    parser.add_argument("-ep", "--epochs", help = "Number of training epochs",
                        type=int, required=False, default=100)
    parser.add_argument("-st", "--steps", help = "Number of learning steps",
                        type=int, required=False, default=100)
    parser.add_argument("-b", "--batch", help = "Batch size",
                        type=int, required=False, default=64)    
    parser.add_argument("-lr", "--learn_rate", help = "Learning rate",
                        type=float, required=False, default=1e-4)
    parser.add_argument("-g", "--gamma", help = "Discount factor",
                        type=float, required=False, default=0.99)
    parser.add_argument("-tr", "--train", help = "Training a new model or continue training from saved model", 
                        choices=['new', 'continue', 'none'], required=False, default='new')
    parser.add_argument("-te", "--test", help = "Testing flag",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-tn", "--num_test", help = "Number of tests",
                        type=int, required=False, default=1)    
    parser.add_argument("-md", "--model_dir", help = "Model directory",
                        type=str, required=False, default=None)
    parser.add_argument("-ms", "--model_step", help = "Model time step",
                        type=str, required=False, default=None)
    
    args = parser.parse_args()

    env_name = args.env
    envs_values = envs.values()
    if env_name not in envs_values:
        print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Environment name does not exist.")
        raise SystemExit(f"Available environments:\n{[value for value in envs_values]}")

    algo_name = args.algo
    algo_values = algos.values()
    if algo_name not in algo_values:
        print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Algorithm not available.")
        raise SystemExit(f"Available algorithms:\n{[value for value in algo_values]}")

    n_env = args.env_num
    seed = args.seed
    n_epochs = args.epochs
    n_steps = args.steps
    batch_size = args.batch
    lr = args.learn_rate
    gamma = args.gamma
    TRAIN = args.train
    TEST = args.test
    n_test = args.num_test

    layers_size = [512, 512, 512]
    max_episode_steps = n_steps * 2
    total_timesteps = n_steps * n_env * 4
    video_length = max_episode_steps

    model_dir = args.model_dir
    model_timestep = args.model_step

    if model_dir is None:
        if TRAIN == 'continue' or (TRAIN == 'none' and TEST):
            parser.error("Valid argument --model_dir must be provided where previous model training files are saved")
        
        else:
            model_time = int(time.time())
            model_name = str(seed*10) + "_" + str(model_time)
            model_dir = f"{models_dir}/{env_name}/{algo_name}/{model_name}"
    else:
        model_name = model_dir.split("/")[-1]
    
    log_dir = f"{logs_dir}/{env_name}/{algo_name}"
    video_dir = f"{videos_dir}/{env_name}/{algo_name}"
    stats_path = f"{log_dir}/{model_name}_0/vec_normalize.pkl"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)


    if algo_name == 'SAC':
        algo = SAC
        n_env = 1
        env = load_environment(env_name, 0, seed*10, max_episode_steps)()
        test_env = env

        if type(env.action_space) != spaces.Box:
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "SAC only supports continuous action spaces.")
            exit(1)

        policy_kwargs = dict(net_arch=dict(pi=layers_size, qf=layers_size))
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=gamma, learning_rate=lr, batch_size=batch_size, ent_coef='auto', learning_starts=100, tensorboard_log=log_dir)
        
        env = Monitor(env, log_dir)

    elif algo_name == 'PPO':
        algo = PPO
        env = SubprocVecEnv([load_environment(env_name, i, seed, max_episode_steps) for i in range(n_env)])
        test_env = load_environment(env_name, 0, seed*10)()

        policy_kwargs = dict(net_arch=dict(pi=layers_size, vf=layers_size))
        model = PPO("MlpPolicy", env, n_steps=n_steps, batch_size=batch_size, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1, learning_rate=lr, tensorboard_log=log_dir)

        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        env = VecMonitor(env, log_dir)

    elif algo_name == 'A2C':
        algo = A2C
        env = SubprocVecEnv([load_environment(env_name, i, seed, max_episode_steps) for i in range(n_env)])
        test_env = load_environment(env_name, 0, seed*10)()

        policy_kwargs = dict(net_arch=dict(pi=layers_size, vf=layers_size))
        model = A2C("MlpPolicy", env, n_steps=n_steps, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1, learning_rate=lr, tensorboard_log=log_dir)
        
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        env = VecMonitor(env, log_dir)

    elif algo_name == 'DQN':
        algo = DQN
        n_env = 1
        env = load_environment(env_name, 0, seed*10, max_episode_steps)()
        test_env = env

        if type(env.action_space) != spaces.Discrete:
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "DQN only supports discrete action spaces.")
            exit(1)

        policy_kwargs = dict(net_arch=layers_size)
        model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=gamma, learning_rate=lr, batch_size=batch_size, learning_starts=100, tensorboard_log=log_dir)
        
        env = Monitor(env, log_dir)

    elif algo_name == 'TD3':
        algo = TD3
        n_env = 1
        env = load_environment(env_name, 0, seed*10, max_episode_steps)()
        test_env = env

        if type(env.action_space) != spaces.Box:
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "TD3 only supports continuous action spaces.")
            exit(1)

        policy_kwargs = dict(net_arch=dict(pi=layers_size, qf=layers_size))
        model = TD3("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=gamma, learning_rate=lr, batch_size=batch_size, learning_starts=100, tensorboard_log=log_dir)
        
        env = Monitor(env, log_dir)

    elif algo_name == 'DDPG':
        algo = DDPG
        n_env = 1
        env = load_environment(env_name, 0, seed*10, max_episode_steps)()
        test_env = env

        if type(env.action_space) != spaces.Box:
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "DDPG only supports continuous action spaces.")
            exit(1)

        policy_kwargs = dict(net_arch=dict(pi=layers_size, qf=layers_size))
        model = DDPG("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=gamma, learning_rate=lr, batch_size=batch_size, learning_starts=100, tensorboard_log=log_dir)
        
        env = Monitor(env, log_dir)

    else:
        print("[ERROR] >> algorithm number is between {1, 6}")
        exit(1)

    model.set_env(env)
    seed = seed*10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.action_space.np_random.seed(seed)

    rewards, final_rewards, steps = [], [], []
    best = -100000
    last_el = 0

    idx = 0
    print("\n-------------------------------")
    print(">>>    Start")
    print("-------------------------------\n")
    start_time = time.time()

    if TRAIN == 'new':
        while idx < n_epochs:
            try:
                print("\n-------------------------------")
                print(">>>    Start training n°", idx+1)
                print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))
                print("[INFO]  >>    scene: ", env_name)
                print("[INFO]  >>    algo: ", algo_name)
                print("[INFO]  >>    seed: ", seed)
                print("-------------------------------\n")

                model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=True, tb_log_name=f"{model_name}", log_interval=1)
                model.save(f"{model_dir}/{total_timesteps*(idx+1)}")

                print("\n-------------------------------")
                print(">>>    Start test n°", idx+1)
                print("[INFO]  >>    scene: ", env_name)
                print("[INFO]  >>    algo: ", algo_name)
                print("[INFO]  >>    seed: ", seed)
                print("-------------------------------\n")

                r, final_r = test(test_env, model, idx, n_steps, n_episode=n_test, render=True)
                final_rewards.append(final_r)
                rewards.append(r)
                steps.append(total_timesteps*(idx+1))

                with open(f"{log_dir}/{model_name}_0/rewards.txt", 'w') as fp:
                    json.dump([rewards, steps], fp)
                with open(f"{log_dir}/{model_name}_0/final_rewards.txt", 'w') as fp:
                    json.dump([final_rewards, steps], fp)

                if r >= best:
                    print(">>>    Save training n°", idx+1)
                    model.save(f"{model_dir}/best")
                    model_timestep = 'best'
                    best = r

                idx+=1

            except:
               print("[ERROR]  >> The simulation failed. Restart from previous id.")
               
            model.save(f"{model_dir}/latest")

        print(">>   End.")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))
    
    elif TRAIN == 'continue':
        if model_timestep is None:
            model_timestep = 'latest'
        save_path = f"{model_dir}/{model_timestep}"

        if not os.path.exists(save_path + ".zip"):
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Model file does not exist")
            exit(1)

        model = algo.load(save_path, tensorboard_log=log_dir)
        model.set_env(env)
        
        print("\n-------------------------------")
        print(f">>>    Continue model training from timestep: {model_timestep}")     

        while idx < n_epochs:
            try:
                print("\n-------------------------------")              
                print(">>>    Start training n°", idx+1)
                print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))
                print("[INFO]  >>    scene: ", env_name)
                print("[INFO]  >>    algo: ", algo_name)
                print("[INFO]  >>    seed: ", seed)
                print("-------------------------------\n")

                model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=True, tb_log_name=f"{model_name}", log_interval=1)
                model.save(f"{model_dir}/{total_timesteps*(idx+1)}")

                print("\n-------------------------------")
                print(">>>    Start test n°", idx+1)
                print("[INFO]  >>    scene: ", env_name)
                print("[INFO]  >>    algo: ", algo_name)
                print("[INFO]  >>    seed: ", seed)
                print("-------------------------------\n")

                r, final_r = test(test_env, model, idx, n_steps, n_episode=n_test, render=True)
                final_rewards.append(final_r)
                rewards.append(r)
                steps.append(total_timesteps*(idx+1))

                with open(f"{log_dir}/{model_name}_0/rewards.txt", 'w') as fp:
                    json.dump([rewards, steps], fp)
                with open(f"{log_dir}/{model_name}_0/final_rewards.txt", 'w') as fp:
                    json.dump([final_rewards, steps], fp)

                if r >= best:
                    print(">>>    Save training n°", idx+1)
                    model.save(f"{model_dir}/best")
                    model_timestep = 'best'
                    best = r

                idx+=1

            except:
               print("[ERROR]  >> The simulation failed. Restart from previous id.")
               
            model.save(f"{model_dir}/latest")

        print(">>   End.")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))

    if TEST:
        del model
        
        if model_timestep is None:
            model_timestep = 'best'
        save_path = f"{model_dir}/{model_timestep}"

        if not os.path.exists(save_path + ".zip"):
            print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Model test file does not exist")
            exit(1)

        model = algo.load(save_path, tensorboard_log=log_dir)
        model.set_env(env)
        
        print("\n-------------------------------")
        print(f">>>    Testing model at timestep: {model_timestep}")
        r, final_r = test(test_env, model, 1, n_steps, n_episode=n_test, render=True)
        print("[INFO]  >>    Best reward : ", r, " - Final reward:", final_r)

    env.close()

    print("... End.")
