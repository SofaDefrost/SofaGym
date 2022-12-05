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

from sofagym import *
from sofagym.envs import *
RANDOM = False

import psutil
pid = os.getpid()
py = psutil.Process(pid)

sys.path.insert(0, os.getcwd()+"/..")

__import__('sofagym')
name = ['multigaitrobot-v0',
        'gripper-v0',       
        'trunk-v0',         
        'trunkcup-v0',      
        'diamondrobot-v0',  
        'maze-v0',          
        'simple_maze-v0',   
        'concentrictuberobot-v0']
num = 7
env_name = name[num]
print("Start env ", env_name)

env = gym.make(env_name)
env.configure({"render":1})
env.configure({"dt":0.01})
env.reset()

env.render()
done = False

strat_multi = [[-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
                            [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
                            [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
                            [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
                            [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0,-1.0, -1.0]]*100
# strat_multi = [[1.0, 1.0, 1.0, 1, 1]]*100


# strat_multi =  [[-1.0, -1.0, -1.0, 1, 1], [1.0, -1.0, -1.0, 1, 1],
#             [1.0, 1.0, 1.0, 1, 1], [1.0, 1.0, 1.0, -1.0, -1.0],
#             [-1.0, 1.0, 1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
#             [-1.0, 1, -1.0, -1.0, 1], [-1.0, -1, 1.0, 1.0, -1],
#             [1.0, 1, -1.0, -1.0, 1], [-1.0, -1.0, 1.0, -1.0, 1],
#             [1.0, -1.0, 1.0, -1.0, -1.0], [-1.0, -1, 1.0, 1.0, -1],
#             [1.0, 1.0, -1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, 1.0, -1],
#             [1.0, -1, 1.0, 1.0, -1]]

start_multi_discrete =  [2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,
                         2, 0, 1, 5, 3, 4,]

strat_jimmy_1 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-0.6, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [-0.6, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0]]+[[0.0, 0.75, 0.75, 0.0, 0.0, 0.0, 1.0, 0.0]]*100

strat_jimmy_0 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                 [-0.8, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]] + [[-0.8, 0.2, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]]*100


print("Start ...")
for i in range(10000000):
    print("\n--------------------------------")
    print("EPISODE - ", i)
    print("--------------------------------\n")
    idx = 0
    tot_reward = 0
    tot_rtf = 0
    done = False
    while not done and idx < 100:
        idx += 1
        #multigaitrobot: [rd.uniform(-1, 1) for i in range(5)] - strat_multi[idx-1]
        #gripper: rd.randint(0,7)
        #trunk: rd.randint(0,15)
        #trunkcup: rd.randint(0,15)
        #diamondrobot: rd.randint(0,7)
        #maze: rd.randint(0,6)
        #simple_maze: rd.randint(0,3)
        #concentrictuberobot: rd.randint(0,11)
        #CartStem [rd.uniform(-1, 1)]
        #action = strat_multi[idx-1]

        action = env.action_space.sample()
        start_time = time.time()
        state, reward, done, _ = env.step(action)
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
