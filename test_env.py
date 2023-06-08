# -*- coding: utf-8 -*-
"""Test the ...Env.

Usage:
-----
    python3.7 test_env.py
"""

__authors__ = ("PSC", "emenager")
__contact__ = ("pierre.schegg@robocath.com", "etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Robocath, Inria"
__date__ = "Oct 7 2020"

import sys
import os
import time
import gym
import argparse
from colorama import Fore

import sofagym
from sofagym.envs import *
RANDOM = False

import psutil
pid = os.getpid()
py = psutil.Process(pid)

sys.path.insert(0, os.getcwd()+"/..")

__import__('sofagym')
name = {
        1:'bubblemotion-v0',
        2:'cartstem-v0',
        3:'cartstemcontact-v0',
        4:'catchtheobject-v0',
        5:'concentrictuberobot-v0',
        6:'diamondrobot-v0',
        7:'gripper-v0',
        8:'maze-v0',
        9:'multigaitrobot-v0',
        10:'simple_maze-v0',
        11:'stempendulum-v0',
        12:'trunk-v0',
        13:'trunkcup-v0',
        14: 'cartpole-v0'
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", help="Name of the environment",
                        type=str, required=True)
    parser.add_argument("-ep", "--episodes", help="Number of episodes",
                        type=int, required=False, default=100)
    parser.add_argument("-s", "--steps", help="Number of steps per episodes",
                        type=int, required=False, default=100)
    args = parser.parse_args()

    env_name = args.env
    episodes = args.episodes
    steps = args.steps

    try:
        env = gym.make(env_name)
    except Exception:
        print(Fore.RED + '[ERROR]   ' + Fore.RESET + "Environment name does not exist.")
        raise SystemExit(f"Available environments:\n{[value for value in name.values()]}")

    print("Start env ", env_name)

    env.configure({"render":1})
    env.configure({"dt":0.01})
    env.reset()

    env.render()
    done = False

    print("Start ...")
    for i in range(episodes):
        print("\n--------------------------------")
        print("EPISODE - ", i+1)
        print("--------------------------------\n")
        idx = 0
        tot_reward = 0
        tot_rtf = 0
        done = False
        while not done and idx < steps:
            idx += 1

            action = env.action_space.sample()

            start_time = time.time()
            state, reward, done, info = env.step(action)
            step_time = time.time()-start_time
            print("[INFO]   >>> Time:", step_time)

            rtf = env.config["dt"]*env.config["scale_factor"]/step_time
            print("[INFO]   >>> RTF:", rtf)

            tot_reward+= reward
            tot_rtf+= rtf

            env.render()

            print("Step ", idx, " action : ", action, " reward : ", reward, " done:", done)

        print("[INFO]   >>> TOTAL REWARD IS:", tot_reward)
        print("[INFO]   >>> FINAL REWARD IS:", reward)
        print("[INFO]   >>> MEAN RTF IS:", tot_rtf/idx)
        memoryUse = py.memory_info()[0]/2.**30
        print("[INFO]   >>> Memory usage:", memoryUse)
        print("[INFO]   >>> Object size:", sys.getsizeof(env))

        env.reset()


    print(">> TOTAL REWARD IS:", tot_reward)
    env.close()
    print("... End.")
