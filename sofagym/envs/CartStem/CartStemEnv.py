# -*- coding: utf-8 -*-
"""Specific environment for the gripper."""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Feb 3 2021"

from sofagym.AbstractEnv import AbstractEnv, EnvWrapper
from sofagym.ServerEnv import ServerEnv

from gymnasium import spaces
import os, sys
import numpy as np

from typing import Optional


class CartStemEnv(EnvWrapper):
    """Sub-class of AbstractEnv, dedicated to the gripper scene.

    See the class AbstractEnv for arguments and methods.
    """

    # Setting a default configuration
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {"render_modes": ["human", "rgb_array"]}
    dim_state = 4
    DEFAULT_CONFIG = {
        "scene": "CartStem",
        "keyboard_controller": False,
        "deterministic": True,
        "source": [0, -70, 10],
        "target": [0, 0, 10],
        "goal": False,
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
        "python_version": sys.version,
        "zFar": 4000,
        "time_before_start": 0,
        "seed": None,
        "init_x": 0,
        "max_move": 40,
        "nb_actions": 2,
        "dim_state": dim_state,
        "randomize_states": False,
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

        self.env.action_space = spaces.Discrete(self.env.nb_actions)
        self.nb_actions = str(self.env.nb_actions)

        low_coordinates = np.array([-100] * self.env.dim_state)
        high_coordinates = np.array([100] * self.env.dim_state)
        self.env.observation_space = spaces.Box(low_coordinates, high_coordinates, dtype=np.float32)

        if self.env.root is None and not self.use_server:
            self.env.init_root()

    def initialize_states(self):
        self.env.initialize_states()

        self.env.config.update(
            {
                "init_x": -(self.env.config["max_move"] / 8)
                + (self.env.config["max_move"] / 4) * self.env.np_random.random()
            }
        )

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if abs(obs[0]) > self.env.config["max_move"]:
            done = True

        return obs, reward, done, truncated, info
