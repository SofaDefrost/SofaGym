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

TRAIN = True
TEST = False


from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from AVEC.stable_baselines import PPO2
from AVEC.stable_baselines.sac import SAC as SAC_AVEC

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

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


# Adapted from rl-agents
def load_environment(id, rank, seed = 0):
    def _init():
        __import__('sofagym')
        env = gym.make(id)
        env.seed(seed + rank)
        env.reset()
        return env

    return _init



def test(env, model, epoch, n_test=1, render = False):
    if render:
        env.config.update({"render":1})
    r, final_r = 0, 0
    for t in range(n_test):
        print("Start >> Epoch", epoch, "- Test", t)
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
                print("Test", t, "- Epoch ", id ,"- Took action: ", action, "- Got reward: ", reward)
                env.render()
            rewards.append(reward)
            id+=1
        print("Done >> Test", t, "- Reward = ", rewards, "- Sum reward:", sum(rewards))
        r+= sum(rewards)
        final_r+= reward
    print("[INFO]  >> Mean reward: ", r/n_test, " - Mean final reward:", final_r/n_test)
    return r/n_test, final_r/n_test

def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "--num_env", help = "Number of the env",
                        type=int, required = True)
    parser.add_argument("-na", "--num_algo", help = "Number of the algorithm",
                        type=int, required = True)
    parser.add_argument("-nc", "--num_cpu", help = "Number of cpu",
                        type=int)
    parser.add_argument("-s", "--seed", help = "The seed",
                        type=int, required = True)
    args = parser.parse_args()

    ids = ['cartstemcontact-v0', 'cartstem-v0', "stempendulum-v0",
            'catchtheobject-v0', "multigaitrobot-v0"]
    timer_limits = [30, 80, 50, 30, 18]
    continues = [True, False, True, True, False]
    n_epochs = [600, 200, 10001, 200, 10001]

    gammas = [0.99, 0.99, 0.99, 0.99, 0.99]
    learning_rates = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
    value_coeffs = [0, 0, 0, 0, 0]
    batch_sizes = [200, 256, 64, 256, 144]
    size_layers = [[512, 512, 512], [512, 512, 512], [512, 512, 512],
                    [512, 512, 512], [512, 512, 512]]


    gamma = gammas[args.num_env]
    learning_rate = learning_rates[args.num_env]
    value_coeff = value_coeffs[args.num_env]
    batch_size = batch_sizes[args.num_env]
    size_layer = size_layers[args.num_env]

    id = ids[args.num_env]
    timer_limit = timer_limits[args.num_env]
    cont = continues[args.num_env]

    if args.num_algo == 0 and cont:
        env = load_environment(id, rank = 0, seed = args.seed*10)()
        test_env = env

        algo = 'SAC'
        policy_kwargs = dict(net_arch=dict(pi=size_layer, qf=size_layer))
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=gamma, learning_rate=learning_rate, batch_size = batch_size, ent_coef='auto', learning_starts=500)
    elif args.num_algo == 1:

        if args.num_cpu is None:
            print("[WARNING] >> Default number of cpu: 4.")
            n_cpu = 4
        else:
            n_cpu = args.num_cpu

        env = SubprocVecEnv([load_environment(id, i, seed = args.seed) for i in range(n_cpu)])
        test_env = load_environment(id, 0, seed = args.seed*10)()
        algo = 'PPO'

        policy_kwargs = dict(net_arch=[dict(pi=size_layer, vf=size_layer)])
        model = PPO("MlpPolicy", env, n_steps=timer_limit*20, batch_size=batch_size, gamma=gamma, policy_kwargs=policy_kwargs, verbose = 1, learning_rate=learning_rate)
    elif args.num_algo == 2:
        env = load_environment(id, rank = 0, seed = args.seed*10)()
        test_env = env

        algo = 'PPO_AVEC'
        policy_kwargs = dict(net_arch=[dict(pi=size_layer, vf=size_layer)])
        model = PPO2('MlpPolicy', env, avec_coef=1., vf_coef=value_coeff, n_steps=timer_limit*20, nminibatches = 40, gamma=gamma, policy_kwargs=policy_kwargs, verbose = 1, learning_rate=learning_rate)
    elif args.num_algo == 3 and cont:
        env = load_environment(id, rank = 0, seed = args.seed*10)()
        test_env = env

        algo = 'SAC_AVEC'
        layers = size_layer
        model = SAC_AVEC('CustomSACPolicy', env, avec_coef=1., value_coef=value_coeff, policy_kwargs={"layers":layers}, verbose=1, gamma=gamma, learning_rate=learning_rate, batch_size = batch_size, ent_coef='auto', learning_starts=500)
    else:
        if not cont and args.num_algo in [0, 3]:
            print("[ERROR] >> SAC is used with continue action space.")
        else:
            print("[ERROR] >> num_algo is in {0, 1, 2, 3}")
        exit(1)

    name = algo + "_" + id + "_" + str(args.seed*10)
    os.makedirs("./Results_benchmark/" + name, exist_ok = True)


    seed = args.seed*10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    #env.seed(seed)
    env.action_space.np_random.seed(seed)

    rewards, final_rewards, steps = [], [], []
    best = -100000
    last_el = 0

    idx = 0
    print("\n-------------------------------")
    print(">>>    Start")
    print("-------------------------------\n")
    start_time = time.time()

    if TRAIN:
        while idx < n_epochs[args.num_env]:

           try:

               print("\n-------------------------------")
               print(">>>    Start training n°", idx+1)
               print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))
               print("[INFO]  >>    scene: ", id)
               print("[INFO]  >>    algo: ", algo)
               print("[INFO]  >>    seed: ", seed)
               print("-------------------------------\n")

               model.learn(total_timesteps=timer_limit*20, log_interval=20)
               model.save("./Results_benchmark/" + name + "/latest")

               print("\n-------------------------------")
               print(">>>    Start test n°", idx+1)
               print("[INFO]  >>    scene: ", id)
               print("[INFO]  >>    algo: ", algo)
               print("[INFO]  >>    seed: ", seed)
               print("-------------------------------\n")

               r, final_r = test(test_env, model, idx, n_test=5)
               final_rewards.append(final_r)
               rewards.append(r)
               steps.append(timer_limit*20*(idx+1))


               with open("./Results_benchmark/" +  name + "/rewards_"+id+".txt", 'w') as fp:
                   json.dump([rewards, steps], fp)
               with open("./Results_benchmark/" +  name + "/final_rewards_"+id+".txt", 'w') as fp:
                   json.dump([final_rewards, steps], fp)

               if r >= best:
                   print(">>>    Save training n°", idx+1)
                   model.save("./Results_benchmark/" +  name + "/best")

               idx+=1
           except:
               print("[ERROR]  >> The simulation failed. Restart from previous id.")

        model.save("./Results_benchmark/" +  name + "/latest")

        print(">>   End.")
        print("[INFO]  >>    time: ", sec_to_hours(time.time()-start_time))

    if TEST:
        del model
        save_path = "./Results_benchmark/" +  name + "/best"

        if args.num_algo == 0 and cont:
            model = SAC.load(save_path)
        elif args.num_algo == 1:
            model = PPO.load(save_path)
        elif args.num_algo == 2:
            model = PPO2.load(save_path)
        elif args.num_algo == 3 and cont:
            model = SAC_AVEC.load(save_path)
        else:
            if not cont and args.num_algo in [0, 3]:
                print("[ERROR] >> SAC is used with continue action space.")
            else:
                print("[ERROR] >> num_algo is in {0, 1, 2, 3}")
            exit(1)

        r, final_r = test(test_env, model, -1, n_test=5, render = True)
        print("[INFO]  >>    Best reward : ", r, " - Final reward:", final_r)
