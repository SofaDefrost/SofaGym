# -*- coding: utf-8 -*-
"""Specific environment for the gripper.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

import os

from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene

from gym import spaces

import numpy as np

class CatchTheObject(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "CatchTheObject",
                      "deterministic": True,
                      "source": [0, -70, 10],
                      "target": [0, 0, 10],
                      "goalList": [[7, 0, 20]],
                      "start_node": None,
                      "scale_factor": 10,
                      "dt": 0.01,
                      "timer_limit": 30,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/CartStem",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": "python3.9",
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "max_move": 10,
                      "max_pressure": 15
                      }

    def __init__(self, config = None):
        super().__init__(config)
        nb_actions = -1
        low = np.array([-1]*1)
        high = np.array([1]*1)
        self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype='float32')
        self.nb_actions = str(nb_actions)

        dim_state = 5
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')

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

        self.count = 0
        self.config.update({'goalPos': self.goal})
        # obs = super().reset()
        # return np.array(obs)
        obs = start_scene(self.config, self.nb_actions)

        return np.array(obs['observation'])

    def get_available_actions(self):
        """Gives the actions available in the environment.

        Parameters:
        ----------
            None.

        Returns:
        -------
            list of the action available in the environment.
        """
        return self.action_space



