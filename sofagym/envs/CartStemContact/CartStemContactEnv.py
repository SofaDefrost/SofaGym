# -*- coding: utf-8 -*-
"""Specific environment for the gripper.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

import os, sys

from sofagym.AbstractEnv import AbstractEnv
from sofagym.ServerEnv import ServerEnv
from sofagym.rpc_server import start_scene

from gym import spaces

import numpy as np

from typing import Optional

class CartStemContactEnv:
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    dim_state = 8
    DEFAULT_CONFIG = {"scene": "CartStemContact",
                      "deterministic": True,
                      "source": [0, -50, 10],
                      "target": [0, 0, 10],
                      "goal": True,
                      "goalList": [[7, 0, 20]],
                      "start_node": None,
                      "scale_factor": 30,
                      "dt": 0.01,
                      "timer_limit": 30,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/CartStemContact",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": sys.version,
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "init_x": 5,
                      "cube_x": [-6, 6],
                      "max_move": 7.5,
                      "nb_actions": -1,
                      "dim_state": dim_state,
                      "init_states": [0] * dim_state,
                      "randomize_states": False,
                      "use_server": False
                      }

    def __init__(self, config = None, root=None, use_server: Optional[bool]=None):
        if use_server is not None:
            self.DEFAULT_CONFIG.update({'use_server': use_server})
        self.use_server = self.DEFAULT_CONFIG["use_server"]
        self.env = ServerEnv(self.DEFAULT_CONFIG, config, root=root) if self.use_server else AbstractEnv(self.DEFAULT_CONFIG, config, root=root)

        self.initialize_states()

        if self.env.config["goal"]:
            self.init_goal()

        low = np.array([-1]*1)
        high = np.array([1]*1)
        self.env.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
        self.nb_actions = str(self.env.nb_actions)

        low_coordinates = np.array([-1]*self.env.dim_state)
        high_coordinates = np.array([1]*self.env.dim_state)
        self.env.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype=np.float32)

        if self.env.root is None and not self.use_server:
            self.env.init_root()

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.env.__getattribute__(name)
    
    def initialize_states(self):
        self.env.initialize_states()
        
        low_cube, high_cube = -6+ 2*self.env.np_random.random(), 6 - 2*self.env.np_random.random()
        self.env.config.update({'cube_x': [low_cube, high_cube]})
        self.env.config.update({'init_x': (low_cube + 3) + (high_cube-low_cube-3)*self.env.np_random.random()})

        if self.env.np_random.random() > 0.5:
            x_goal = low_cube + 3.5*self.env.np_random.random()
        else:
            x_goal = high_cube - 3.5*self.env.np_random.random()

        self.env.config.update({'goalList': [[x_goal, 0, 20]]})
        self.env.config.update({'max_move': max(abs(low_cube-1), high_cube+1)})

    def reset(self):
        """Reset simulation.
        """
        self.initialize_states()

        if self.env.config["goal"]:
            self.init_goal()

        self.env.reset()

        if self.use_server:
            obs = start_scene(self.env.config, self.nb_actions)
            state = np.array(obs['observation'], dtype=np.float32)
        else:
            state = np.array(self.env._getState(self.env.root), dtype=np.float32)
        
        return state
