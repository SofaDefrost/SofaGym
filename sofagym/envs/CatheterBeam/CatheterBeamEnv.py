import os
import sys

import numpy as np
from gym import spaces

from sofagym.AbstractEnv import AbstractEnv
from sofagym.rpc_server import start_scene


class CatheterBeamEnv(AbstractEnv):
    """Sub-class of AbstractEnv, dedicated to the catheter beam scene.

    See the class AbstractEnv for arguments and methods.
    """
    #Setting a default configuration
    path = path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    DEFAULT_CONFIG = {"scene": "CatheterBeam",
                      "deterministic": True,
                      "source": [-1169.51, 298.574, 257.631],
                      "target": [0, 0, 0],
                      "start_node": None,
                      "scale_factor": 10,
                      "dt": 0.01,
                      "timer_limit": 80,
                      "timeout": 50,
                      "display_size": (1600, 800),
                      "render": 0,
                      "save_data": False,
                      "save_image": False,
                      "save_path": path + "/Results" + "/CatheterBeam",
                      "planning": False,
                      "discrete": False,
                      "start_from_history": None,
                      "python_version": sys.version,
                      "zFar": 4000,
                      "time_before_start": 0,
                      "seed": None,
                      "scale": 30,
                      "rotation": [140.0, 0.0, 0.0],
                      "translation": [0.0, 0.0, 0.0],
                      "goalList": [1226, 1663, 1797, 1544, 2233, 2580, 3214]
                      }

    def __init__(self, config = None):
        super().__init__(config)
        nb_actions = 12
        self.action_space = spaces.Discrete(nb_actions)
        self.nb_actions = str(nb_actions)

        dim_state = 12
        low_coordinates = np.array([-1]*dim_state)
        high_coordinates = np.array([1]*dim_state)
        self.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype=np.float32)

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
