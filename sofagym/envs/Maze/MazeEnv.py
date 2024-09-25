# -*- coding: utf-8 -*-
"""Specific environment for the trunk (simplified).
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import os, sys
import numpy as np

from sofagym.AbstractEnv import AbstractEnv, ServerEnv
from sofagym.rpc_server import start_scene

from gym import spaces

from typing import Optional

class MazeEnv:
    """Sub-class of AbstractEnv, dedicated to the trunk scene.

    See the class AbstractEnv for arguments and methods.
    """
    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    dim_state = 9
    DEFAULT_CONFIG = {"scene": "Maze",
                      "deterministic": True,
                      "source": [-82.0819, 186.518, 135.963],
                      "target": [-2.09447, 5.75347, -4.34572],
                      "goal": True,
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
                      "python_version": sys.version,
                      "zFar": 1000,
                      "dt": 0.01,
                      "time_before_start": 20,
                      "nb_actions": 6,
                      "dim_state": dim_state,
                      "randomize_states": False,
                      "init_states": [0] * dim_state,
                      "use_server": False
                      }

    def __init__(self, config = None, root=None, use_server: Optional[bool]=False):
        self.use_server = self.DEFAULT_CONFIG["use_server"]
        self.env = ServerEnv(self.DEFAULT_CONFIG, config, root=root) if self.use_server else AbstractEnv(self.DEFAULT_CONFIG, config, root=root)

        self.initialize_states()

        if self.env.config["goal"]:
            self.init_goal()

        nb_actions = self.env.config["nb_actions"]
        self.env.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = self.env.config["dim_state"]
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.env.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype=np.float32)

        if self.env.root is None and not self.use_server:
            self.env.init_root()

    # called when an attribute is not found:
    def __getattr__(self, name):
        # assume it is implemented by self.instance
        return self.env.__getattribute__(name)

    def initialize_states(self):
        if self.env.config["randomize_states"]:
            self.init_states = self.randomize_init_states()
            self.env.config.update({'init_states': list(self.init_states)})
        else:
            self.init_states = self.env.config["init_states"]

    def randomize_init_states(self):
        """Randomize initial states.

        Returns:
        -------
            init_states: list
                List of random initial states for the environment.
        
        Note:
        ----
            This method should be implemented according to needed random initialization.
        """
        return self.env.config["init_states"]

    def init_goal(self):
        # Set a new random goal from the list
        goalList = self.env.config["goalList"]
        id_goal = self.env.np_random.choice(range(len(goalList)))
        self.env.config.update({'goal_node': id_goal})
        self.env.goal = goalList[id_goal]
        self.env.config.update({'goalPos': self.env.goal})

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
        return self.env.action_space
