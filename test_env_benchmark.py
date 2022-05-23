# -*- coding: utf-8 -*-
"""Test the ...Env.

Usage:
-----
    python3.8 test_env_benchmark.py
"""

__authors__ = ("PSC", "emenager")
__contact__ = ("pierre.schegg@robocath.com", "etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Robocath, Inria"
__date__ = "Oct 7 2020"

import sys
import os
import time
import random as rd
import gym

import psutil
pid = os.getpid()
py = psutil.Process(pid)

RANDOM = False

sys.path.insert(0, os.getcwd()+"/..")

def load_environment(idx, rank=0, seed = 0):
    def _init():
        __import__('sofagym')
        print(idx)
        env = gym.make(idx)
        env.seed(seed + rank)
        env.reset()
        return env

    return _init


name = ['cartstemcontact-v0', 'cartstem-v0', "stempendulum-v0",
        "catchtheobject-v0", "cartstemcontact-v2"]
isContinue = [True, False, True, True, True]
dim = [1, 2, 1, 1, 1]
num = 0

env_name = name[num]
env_continue = isContinue[num]
env_dim = dim[num]
print("Start env ", env_name)

env = load_environment(env_name, rank =0, seed = 0)()
env.configure({"render":1})
env.reset()
env.render()

print("Start ...")
for i in range(10000000):
    print("\n--------------------------------")
    print("EPISODE - ", i)
    print("--------------------------------\n")
    idx = 0
    tot_reward = 0
    tot_rtf = 0
    done = False
    while not done and idx < 275:
        idx += 1
        if env_continue:
            action = [rd.uniform(-1, 1) for _ in range(env_dim)]
        else:
            action = rd.randint(0,env_dim)

        start_time = time.time()
        state, reward, done, _ = env.step(action)
        step_time = time.time()-start_time
        print("[INFO]   >>> Time:", step_time)
        rtf = env.config["dt"]*env.config["scale_factor"]/step_time
        print("[INFO]   >>> RTF:", rtf)
        tot_reward+= reward
        tot_rtf+= rtf
        env.render()

        print("Step ", idx, " done : ", done,  " state : ", state, " reward : ", reward)
    print("[INFO]   >>> TOTAL REWARD IS:", tot_reward)
    print("[INFO]   >>> MEAN RTF IS:", tot_rtf/idx)
    memoryUse = py.memory_info()[0]/2.**30
    print("[INFO]   >>> Memory usage:", memoryUse)
    print("[INFO]   >>> Object size:", sys.getsizeof(env))

    env.reset()
env.close()
print("... End.")
