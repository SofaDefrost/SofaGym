import math
import os
import sys

import numpy as np
from gym import spaces
from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene


class CartPoleEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the cart pole scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "CartPole",
                      "deterministic": True,
                      "source": [0, 0, 160],
                      "target": [0, 0, 0],
                      "goalList": [[0]],
                      "start_node": None,
                      "scale_factor": 10,
                      "dt": 0.001,
                      "timer_limit": 80,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/CartPole",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": sys.version,
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "init_x": 0,
                      "max_move": 24,
                      }

    def __init__(self, config=None):
        super().__init__(config)

        self.x_threshold = 100
        self.theta_threshold_radians = self.config["max_move"] * math.pi / 180
        self.config.update({'max_angle': self.theta_threshold_radians})
        
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        nb_actions = 2
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        """Reset simulation.

        Note:
        ----
            We launch a client to create the scene. The scene of the program is
            client_<scene>Env.py.

        """
        super().reset()

        self.config.update({'goalPos': self.goal})
        init_states = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.config.update({'init_states': list(init_states)})
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
