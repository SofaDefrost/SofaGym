# -*- coding: utf-8 -*-
"""Specific environment for the Diamond Robot.
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 01 2021"

import os, sys

from sofagym.AbstractEnv import AbstractEnv, ServerEnv
from sofagym.rpc_server import start_scene
from sofagym.viewer import LegacyViewer
from sofagym.envs.Diamond.DiamondToolbox import startCmd

from gym import spaces

import numpy as np

from typing import Optional

class DiamondRobotEnv:
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    dim_state = (6, 3)
    DEFAULT_CONFIG = {"scene": "Diamond",
                      "deterministic": True,
                      "source": [-288, -81, 147],
                      "target": [4, -6, 52],
                      "goal": True,
                      "goalList": [[30.0, 0.0, 150.0], [-30.0, 0.0, 150.0], [0.0, 30.0, 150.0], [0.0, -30.0, 150.0]],
                      "scale_factor": 5,
                      "timer_limit": 50,
                      "timeout": 30,
                      "display_size": (1600, 800),
                      "render": 1,
                      "save_data": True,
                      "save_path": path + "/Results" + "/Diamond",
                      "planning": True,
                      "discrete": True,
                      "seed": 0,
                      "start_from_history": None,
                      "python_version": sys.version,
                      "zFar": 5000,
                      "dt": 0.01,
                      "nb_actions": 8,
                      "dim_state": dim_state,
                      "randomize_states": False,
                      "init_states": 0,
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

        low_coordinates = np.ones(shape=self.env.dim_state)*-1
        high_coordinates = np.ones(shape=self.env.dim_state)
        self.env.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype=np.float32)

        if self.env.root is None and not self.use_server:
            self.env.init_root()

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.env.__getattribute__(name)

    def init_goal(self):
        # Set a new random goal from the list
        self.env.goal = [-30 + 60 * self.env.np_random.random(), -30 + 60 * self.env.np_random.random(), 125 + 20 * self.env.np_random.random()]
        self.env.config.update({'goalPos': self.env.goal})

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
