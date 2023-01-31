# -*- coding: utf-8 -*-
"""Specific environment for the gripper.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene

from gym import spaces
import os
import numpy as np

class CartStemEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "CartStem",
                      "deterministic": True,
                      "source": [0, -70, 10],
                      "target": [0, 0, 10],
                      "goalList": [[7, 0, 20]],
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
                      "python_version": "python3.9",
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "init_x": 0,
                      "max_move": 40,
                      }

    def __init__(self, config = None):
        super().__init__(config)
        nb_actions = 2
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = 4
        low_coordinates = np.array([-100]*dim_state)
        high_coordinates = np.array([100]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates,
                                            dtype='float32')


    def step(self, action):
        obs, reward, done, info = super().step(action)
        if abs(obs[0]) > self.config["max_move"]:
            done = True

        return obs, reward, done, info

    def reset(self):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()

        self.config.update({'init_x': -(self.config["max_move"]/8) + (self.config["max_move"]/4)*np.random.random()})
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
        return self.action_space


