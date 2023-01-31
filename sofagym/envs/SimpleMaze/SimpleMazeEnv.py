# -*- coding: utf-8 -*-
"""Specific environment for the trunk (simplified).
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Mar 23 2021"

import os
import numpy as np

from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene
from sofagym.viewer import LegacyViewer
from sofagym.envs.SimpleMaze.SimpleMazeToolbox import startCmd

from gym import spaces

class SimpleMazeEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "SimpleMaze",
                      "deterministic": True,
                      "source": [0, 200, 0],
                      "target": [0, 0, 0],
                      "goalList": [301, 334, 317, 312, 301],
                      "goal_node": 334,
                      "start_node": 269,
                      "scale_factor": 200,
                      "timer_limit": 50,
                      "timeout": 30,
                      "display_size": (1600, 800),
                      "render": 1,
                      "save_data": False,
                      "save_path": path + "/Results" + "/SimpleMaze",
                      "planning": True,
                      "discrete": True,
                      "seed": 0,
                      "start_from_history": None,
                      "python_version": "python3.9",
                      "zFar": 5000,
                      }

    def __init__(self, config=None):
        super().__init__(config)
        nb_actions = 4
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = 13
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
        if self.nb_actions == "4":
            if not self.past_actions:
                return [0, 1, 2, 3]
            last_action = self.past_actions[-1]
            print(last_action)
            available_actions = [[0, 1, 2],
                                 [0, 1, 3],
                                 [0, 2, 3],
                                 [1, 2, 3]]
            return available_actions[last_action]
        return list(range(int(self.nb_actions)))


