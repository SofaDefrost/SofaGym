import math
import os
import sys
from typing import Optional

import numpy as np
from gymnasium import spaces
from sofagym.AbstractEnv import AbstractEnv, EnvWrapper
from sofagym.ServerEnv import ServerEnv


class CartPoleEnv(EnvWrapper):
    """Sub-class of AbstractEnv, dedicated to the cart pole scene.

    See the class AbstractEnv for arguments and methods.
    """

    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {"render_modes": ["human", "rgb_array"]}
    dim_state = 4
    DEFAULT_CONFIG = {
        "scene": "CartPole",
        "deterministic": True,
        "source": [0, 0, 160],
        "target": [0, 0, 0],
        "goal": False,
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
        "nb_actions": 2,
        "dim_state": dim_state,
        "init_x": 0,
        "x_threshold": 100,
        "max_move": 12,
        "max_angle": 0.418,
        "randomize_states": True,
        "init_states": [0] * dim_state,
        "use_server": False,
    }

    def __init__(self, config=None, root=None, use_server: Optional[bool] = None, render_mode: Optional[str] = None):
        # Not written in DEFAULT_CONFIG, which the class shares: the next envs would use the server too.
        self.use_server = self.DEFAULT_CONFIG["use_server"] if use_server is None else use_server
        config = {**(config or {}), "use_server": self.use_server}
        self.env = (
            ServerEnv(self.DEFAULT_CONFIG, config, render_mode=render_mode, root=root)
            if self.use_server
            else AbstractEnv(self.DEFAULT_CONFIG, config, render_mode=render_mode, root=root)
        )

        self.initialize_states()

        if self.env.config["goal"]:
            self.init_goal()

        self.x_threshold = self.env.config["x_threshold"]
        self.theta_threshold_radians = self.env.config["max_move"] * math.pi / 180
        self.env.config.update({"max_angle": self.theta_threshold_radians})

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.env.action_space = spaces.Discrete(self.env.nb_actions)
        self.nb_actions = str(self.env.nb_actions)

        self.env.observation_space = spaces.Box(-high, high, dtype=np.float32)

        if self.env.root is None and not self.use_server:
            self.env.init_root()

    def initialize_states(self):
        if self.env.config["randomize_states"]:
            self.init_states = self.randomize_init_states()
            self.env.config.update({"init_states": list(self.init_states)})
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
        init_states = self.env.np_random.uniform(low=-0.05, high=0.05, size=(self.env.config["dim_state"],))

        return init_states
