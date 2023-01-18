# -*- coding: utf-8 -*-
"""Specific environment for the Diamond Robot.
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 01 2021"

import os

from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene
from sofagym.viewer import LegacyViewer
from sofagym.envs.Diamond.DiamondToolbox import startCmd

from gym import spaces

import numpy as np

class DiamondRobotEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "Diamond",
                      "deterministic": True,
                      "source": [-288, -81, 147],
                      "target": [4, -6, 52],
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
                      "python_version": "python3.8",
                      "zFar": 5000,
                      }

    def __init__(self, config=None):
        super().__init__(config)
        nb_actions = 8
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = 5
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')

    def step(self, action):
        if self.viewer:
            self.viewer.step(action)

        return super().step(action)

    def reset(self):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()

        self.goal = [-30 + 60 * np.random.random(), -30 + 60 * np.random.random(), 125 + 20 * np.random.random()]

        self.config.update({'goalPos': self.goal})
        obs = start_scene(self.config, self.nb_actions)
        if self.viewer:
            self.viewer.reset()
        self.render()

        return obs['observation']

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
        if not self.viewer:
            display_size = self.config["display_size"]  # Sim display
            self.viewer = LegacyViewer(self, display_size, startCmd=startCmd)

        # Use the viewer to display the environment.
        self.viewer.render()

    def get_available_actions(self):
        """Gives the actions available in the environment.

        Parameters:
        ----------
            None.

        Returns:
        -------
            list of the action available in the environment.
        """
        return list(range(int(self.nb_actions)))
