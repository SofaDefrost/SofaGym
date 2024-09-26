# -*- coding: utf-8 -*-
"""Specific environment for the trunk (simplified).
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Mar 23 2021"

import os, sys
import numpy as np

from sofagym.AbstractEnv import AbstractEnv, ServerEnv
from sofagym.rpc_server import start_scene

from gym import spaces

from typing import Optional

class SimpleMazeEnv:
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    dim_state = 13
    DEFAULT_CONFIG = {"scene": "SimpleMaze",
                      "deterministic": True,
                      "source": [0, 200, 0],
                      "target": [0, 0, 0],
                      "goal": True,
                      "goalList": [301, 334, 317, 312],
                      "goal_node": 334,
                      "start_node": 269,
                      "scale_factor": 10,
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
                      "python_version": sys.version,
                      "zFar": 5000,
                      "dt": 0.01,
                      "nb_actions": 4,
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

        low_coordinates = np.array([-1]*self.env.dim_state)
        high_coordinates = np.array([1]*self.env.dim_state)
        self.env.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype=np.float32)

        if self.env.root is None and not self.use_server:
            self.env.init_root()

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.env.__getattribute__(name)

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
            if not self.env.past_actions:
                return [0, 1, 2, 3]
            last_action = self.env.past_actions[-1]
            print(last_action)
            available_actions = [[0, 1, 2],
                                 [0, 1, 3],
                                 [0, 2, 3],
                                 [1, 2, 3]]
            return available_actions[last_action]
        return self.env.action_space
