# -*- coding: utf-8 -*-
"""Specific environment for the Concentric Tube Robot.
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 01 2021"

from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene
from sofagym.viewer import LegacyViewer
from sofagym.envs.CTR.CTRToolbox import startCmd

from gym import spaces
import os
import numpy as np


class ConcentricTubeRobotEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "CTR",
                      "deterministic": True,
                      "source": [-150, 0, 30],
                      "target": [0, 0, 30],
                      "mesh": "mesh/nasal_cavity.obj",
                      "scale": 30,
                      "rotation": [140.0, 0.0, 0.0],
                      "translation": [0.0, 0.0, 0.0],
                      "goalList": [[0.0, 0.0, 0.0]],
                      "goalPos": [0.0, 0.0, 0.0],
                      "scale_factor": 10,
                      "timer_limit": 50,
                      "timeout": 3,
                      "display_size": (1600, 800),
                      "render": 1,
                      "save_data": True,
                      "save_path": path + "/Results" + "/CTR",
                      "planning": True,
                      "discrete": True,
                      "seed": 0,
                      "start_from_history": None,
                      "python_version": "python3.8",
                      "zFar": 5000,
                      }

    def __init__(self, config=None):
        super().__init__(config)
        nb_actions = 12
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = 1
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')

        self.default_action = 3

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

        y = -20 + 50 * np.random.random()

        self.goal = [0.0, y, abs(y) + 30 * np.random.random()]
        # self.goal = [0.0, 50, 65]
        # self.goal = [0.0, 20, 70]
        # self.goal = [0.0, 30, 35]
        # self.goal = [0.0, -7, 45]

        self.config.update({'goalPos': self.goal})
        print(self.config)
        obs = start_scene(self.config, self.nb_actions)
        if self.viewer:
            self.viewer.reset()
        self.render()

        self.step(0)
        self.render()
        self.step(4)
        self.render()
        self.step(8)
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


