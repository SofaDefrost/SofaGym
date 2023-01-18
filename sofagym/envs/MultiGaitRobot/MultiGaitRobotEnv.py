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

class MultiGaitRobotEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "MultiGaitRobot",
                      "deterministic": True,
                      "goalList": [[250, 0, 0], [-100, 0, 0]],
                      "source": [150.0, -500, 150],
                      "target": [150, 0, 0],
                      "start_node": None,
                      "scale_factor": 60,
                      "timer_limit": 6,
                      "timeout": 200,
                      "dt": 0.01,
                      "display_size": (1600, 800),
                      "render": 2,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/MultiGaitRobot",
                      "planning": True,
                      "discrete": True,
                      "seed": None,
                      "start_from_history": None,
                      "python_version": "python3.9"
                      }

    def __init__(self, config=None):
        super().__init__(config)

        if self.config['discrete']:
            # discrete
            nb_actions = 6
            self.action_space = spaces.Discrete(nb_actions)
            self.nb_actions = str(nb_actions)
        else:
            # Continuous
            nb_actions = -1
            low_coordinates = np.array([-1]*3)
            high_coordinates = np.array([1]*3)
            self.action_space = spaces.Box(low_coordinates, high_coordinates,
                                           dtype='float32')
        self.nb_actions = str(nb_actions)

        dim_state = 32
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
        return list(range(int(self.nb_actions)))


