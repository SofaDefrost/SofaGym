# -*- coding: utf-8 -*-
"""Specific environment for the trunk (simplified).
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import os
import numpy as np

from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene

from gym import spaces

class MazeEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "Maze",
                      "deterministic": True,
                      "source": [0, 200, 0],
                      "target": [0, 0, 0],
                      "goalList": [334, 317, 312, 301],
                      "goal_node": 270,
                      "start_node": 269,
                      "scale_factor": 5,
                      "timer_limit": 250,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 1,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/Maze",
                      "planning": True,
                      "discrete": True,
                      "seed": 0,
                      "start_from_history": None,
                      "python_version": "python3.8",
                      "zFar": 1000
                      }

    def __init__(self, config=None):
        super().__init__(config)
        nb_actions = 6
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = 13
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype='float32')

    def step(self, action):
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

        return obs['observation']

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