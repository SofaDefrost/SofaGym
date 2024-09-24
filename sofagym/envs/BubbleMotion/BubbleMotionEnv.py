# -*- coding: utf-8 -*-
"""Specific environment for the gripper.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

from sofagym.AbstractEnv import AbstractEnv, ServerEnv
from sofagym.rpc_server import start_scene

from gym import spaces
import os, sys
import numpy as np

from typing import Optional

class BubbleMotionEnv:
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "BubbleMotion",
                      "deterministic": True,
                      "source": [5, -5, 20],
                      "target": [5, 5, 0],
                      "goal": True,
                      "goalList": [[7, 0, 20]],
                      "start_node": None,
                      "scale_factor": 20,
                      "timer_limit": 20,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/StemPendulum",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": sys.version,
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "dt": 0.01,
                      "max_pressure": 40,
                      "board_dim": 8,
                      "randomize_states": False,
                      "use_server": False
                      }
    
    def __init__(self, config = None, root=None, use_server: Optional[bool]=False):
        self.use_server = self.DEFAULT_CONFIG["use_server"]
        self.env = ServerEnv(self.DEFAULT_CONFIG, config, root=root) if self.use_server else AbstractEnv(self.DEFAULT_CONFIG, config, root=root)

        nb_actions = -1
        low = np.array([-1]*9)
        high = np.array([1]*9)
        self.env.action_space = spaces.Box(low=low, high=high, shape=(9,), dtype=np.float32)
        self.nb_actions = str(nb_actions)

        dim_state = 15
        low_coordinates = np.array([0]*dim_state)
        high_coordinates = np.array([80]*dim_state)
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
        return self.env.action_space
