# -*- coding: utf-8 -*-
"""Specific environment for the gripper.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

from sofagym.AbstractEnv import AbstractEnv
from sofagym.ServerEnv import ServerEnv
from sofagym.rpc_server import start_scene

from gym import spaces
import os, sys
import numpy as np

from typing import Optional

class CartStemEnv:
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    dim_state = 4
    DEFAULT_CONFIG = {"scene": "CartStem",
                      "deterministic": True,
                      "source": [0, -70, 10],
                      "target": [0, 0, 10],
                      "goal": False,
                      "start_node": None,
                      "scale_factor": 10,
                      "dt": 0.01,
                      "timer_limit": 80,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/CartStem",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": sys.version,
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "init_x": 0,
                      "max_move": 40,
                      "nb_actions": 2,
                      "dim_state": dim_state,
                      "randomize_states": False,
                      "init_states": [0] * dim_state,
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

        self.env.action_space = spaces.Discrete(self.env.nb_actions)
        self.nb_actions = str(self.env.nb_actions)

        low_coordinates = np.array([-100]*self.env.dim_state)
        high_coordinates = np.array([100]*self.env.dim_state)
        self.env.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype=np.float32)

        if self.env.root is None and not self.use_server:
            self.env.init_root()

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.env.__getattribute__(name)
    
    def initialize_states(self):
        self.env.initialize_states()

        self.env.config.update({'init_x': -(self.env.config["max_move"]/8) + (self.env.config["max_move"]/4)*self.env.np_random.random()})

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if abs(obs[0]) > self.env.config["max_move"]:
            done = True

        return obs, reward, done, info

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
