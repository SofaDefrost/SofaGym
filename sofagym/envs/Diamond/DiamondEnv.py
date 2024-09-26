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
                      "use_server": False
                      }

    def __init__(self, config = None, root=None, use_server: Optional[bool]=False):
        self.use_server = self.DEFAULT_CONFIG["use_server"]
        self.env = ServerEnv(self.DEFAULT_CONFIG, config, root=root) if self.use_server else AbstractEnv(self.DEFAULT_CONFIG, config, root=root)

        nb_actions = self.env.config["nb_actions"]
        self.env.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = self.env.config["dim_state"]
        low_coordinates = np.ones(shape=dim_state)*-1
        high_coordinates = np.ones(shape=dim_state)
        self.env.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype=np.float32)

        if self.env.root is None and not self.use_server:
            self.env.init_root()

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.env.__getattribute__(name)

    def step(self, action):
        if self.use_server:
            if self.env.viewer:
                self.env.viewer.step(action)

        return self.env.step(action)

    def reset(self):
        """Reset simulation.
        """
        self.env.reset()

        self.env.goal = [-30 + 60 * self.env.np_random.random(), -30 + 60 * self.env.np_random.random(), 125 + 20 * self.env.np_random.random()]

        self.env.config.update({'goalPos': self.env.goal})

        if self.use_server:
            obs = start_scene(self.env.config, self.nb_actions)
            if self.env.viewer:
                self.env.viewer.reset()
            state = np.array(obs['observation'], dtype=np.float32)
        else:
            state = np.array(self.env._getState(self.env.root), dtype=np.float32)
        
        return state
    
    def render(self, mode='rgb_array'):
        """See the current state of the environment.

        Get the OpenGL Context to render an image (snapshot) of the simulation
        state.

        Parameters:
        ----------
          mode: string, default = 'rgb_array'
              Type of representation.

        Returns:
        -------
          None.
        """
        if self.use_server:
            if not self.env.viewer:
                display_size = self.env.config["display_size"]  # Sim display
                self.env.viewer = LegacyViewer(self, display_size, startCmd=startCmd)

            # Use the viewer to display the environment.
            self.env.viewer.render()
        else:
            self.env.render(mode)

    def get_available_actions(self):
        """Gives the actions available in the environment.

        Parameters:
        ----------
            None.

        Returns:
        -------
            list of the action available in the environment.
        """
        return self.env.action_space
