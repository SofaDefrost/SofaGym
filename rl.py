# -*- coding: utf-8 -*-
"""Train SOFA scenes as gym environments using RL algorithms.

Usage:
-----
    python3 rl.py -e env_id -a algo_name
"""

__authors__ = "emenager, pschegg"
__contact__ = "etienne.menager@ens-rennes.fr, pierre.schegg@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020,Inria"
__date__ = "Nov 10 2020"


import argparse

from agents.RLberryAgent import RLberryAgent
from agents.SB3Agent import SB3Agent

from agents.utils import args_check

import sofagym
from sofagym.envs import *


results_dir = "./Results"

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
        13: 'trunkcup-v0'
        }

algos = {
        1: 'SAC',
        2: 'PPO',
        3: 'A2C',
        4: 'DQN',
        5: 'TD3',
        6: 'DDPG',
        7: 'REINFORCE'
        }

frameworks = {
        1: 'SB3',
        2: 'RLberry'
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help="Name of the environment",
                        type=str, required=True)
    parser.add_argument("-a", "--algorithm", help = "RL algorithm",
                        type=str, required=True)
    parser.add_argument("-fr", "--framework", help = "RL framework",
                        type=str, required=False, default='SB3')
    parser.add_argument("-ne", "--env_num", help = "Number of parallel envs",
                        type=int, required=False, default=1)
    parser.add_argument("-s", "--seed", help = "Seed",
                        type=int, required=False, default=0)
    parser.add_argument("-ep", "--epochs", help = "Number of training epochs",
                        type=int, required=False, default=100)
    parser.add_argument("-mst", "--max_steps", help = "Max steps per episode",
                        type=int, required=False, default=None)
    parser.add_argument("-tr", "--train", help = "Training a new model or continue training from saved model", 
                        choices=['new', 'continue', 'none'], required=False, default='new')
    parser.add_argument("-te", "--test", help = "Testing flag",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-tn", "--num_test", help = "Number of tests",
                        type=int, required=False, default=1)    
    parser.add_argument("-md", "--model_dir", help = "Model directory",
                        type=str, required=False, default=None)
    
    args = parser.parse_args()

    env_name = args.environment
    args_check(env_name, envs, 'environment')
    algo_name = args.algorithm
    args_check(algo_name, algos, 'algorithm')
    framework = args.framework
    args_check(framework, frameworks, 'framework')

    n_envs = args.env_num
    seed = args.seed
    n_epochs = args.epochs
    max_episode_steps = args.max_steps
    train = args.train
    test = args.test
    n_tests = args.num_test
    model_dir = args.model_dir
    
    if model_dir is None:
        if train == 'continue' or (train == 'none' and test):
            parser.error("Valid argument --model_dir must be provided where previous model training files are saved")
    
    Agent = eval(framework + "Agent")

    if train == 'new':
        agent = Agent(env_name, algo_name, seed, results_dir, max_episode_steps, n_envs)
        agent.fit(n_epochs)
    else:
        agent = Agent.load(model_dir)
        
        if train == 'continue':
            agent.fit_continue(n_epochs)

    if test:
        agent.eval(n_tests)

    agent.close()
    print("... End.")
