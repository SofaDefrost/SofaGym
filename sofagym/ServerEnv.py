# -*- coding: utf-8 -*-
"""ServerEnv to use the server-client architecture."""

__authors__ = ("PSC", "dmarchal", "emenager")
__contact__ = ("pierre.schegg@robocath.com", "damien.marchal@univ-lille.fr", "etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Robocath, CNRS, Inria"
__date__ = "Oct 7 2020"

from typing import Optional

import numpy as np
import copy

import importlib

from sofagym.rpc_server import start_server, add_new_step, get_result, clean_registry, close_scene

from sofagym.AbstractEnv import AbstractEnv


class ServerEnv(AbstractEnv):
    def __init__(self, default_config, config=None, render_mode: Optional[str] = None, root=None):
        super().__init__(default_config, config, render_mode=render_mode, root=root)

        # Start the server which distributes the calculations to its clients
        start_server(self.config)

    def step(self, action):
        """Executes one action in the environment.

        Apply action and execute scale_factor simulation steps of 0.01 s.

        Parameters:
        ----------
            action: int
                Action applied in the environment.

        Returns:
        -------
            obs(ObsType):
                The new state of the agent.
            reward(float):
                The reward obtain after applying the action in the current state.
            terminated(bool):
                Whether the agent reaches the terminal state
            truncated(bool):
                Whether the episode reached timer_limit steps
            info(dict):
                additional information (not used here)
        """
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        action = self._formataction(action)

        # Pass the actions to the server to launch the simulation.
        result_id = add_new_step(self.past_actions, action)
        self.past_actions.append(action)

        # Request results from the server.
        # print("[INFO]   >>> Result id:", result_id)
        results = get_result(result_id, timeout=self.timeout)

        obs = np.array(results["observation"], dtype=np.float32)  # to work with baseline
        reward = results["reward"]
        done = results["done"]

        # Avoid long explorations by using a timer: the depth of the state, the steps of a planner explore branches.
        self.timer += 1
        truncated = len(self.past_actions) >= self.config["timer_limit"]

        info = {}  # (not use here)

        # In planning, the steps of the planner explore branches: the agent calls clean() after its real actions.
        return obs, reward, done, truncated, info

    def async_step(self, action):
        """Executes one action in the environment.

        Apply action and execute scale_factor simulation steps of 0.01 s.
        Like step but useful if you want to parallelise (blocking "get").
        Otherwise use step.

        Parameters:
        ----------
            action: int
                Action applied in the environment.

        Returns:
        -------
            LateResult:
                Class which allows to store the id of the client who performs
                the calculation and to return later the usual information
                (observation, reward, done) thanks to a get method.

        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        result_id = add_new_step(self.past_actions, action)
        self.past_actions.append(action)

        class LateResult:
            def __init__(self, result_id):
                self.result_id = result_id

            def get(self, timeout=None):
                results = get_result(self.result_id, timeout=timeout)
                obs = results["observation"]
                reward = results["reward"]
                done = results["done"]
                return obs, reward, done, False, {}

        return LateResult(copy.copy(result_id))

    def reset(self, seed=None, options=None):
        """Reset simulation.

        The env starts the first client of the server with its config (start_scene), which gives the observation.

        Parameters:
        ----------
            seed: int or None, default = None
                Seed of the random generator of the env.
            options: dict or None, default = None
                Not used.

        Returns:
        -------
            None.
        """
        if seed is not None:
            self.seed(seed)
        self.clean()
        self.viewer = None

        # splib3.animation.animate is shadowed by the animate function, reset the global of the module itself.
        importlib.import_module("splib3.animation.animate").manager = None

        self.timer = 0
        self.past_actions = []
        # Before the env starts the first client with its config.
        self.draw_scene_seed()

        return

    def clean(self):
        """Function to clean the registery .

        Close clients who are processing unused sequences of actions (for
        planning): the clients outside the subtree of past_actions, but the
        root. A planning agent calls it after each of its real actions, which
        make the other branches unreachable.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.
        """
        clean_registry(self.past_actions)

    def close(self):
        """Terminate simulation.

        Close the viewer and the scene.

        Parametres:
        ----------
            None.

        Returns:
        -------
            None.
        """
        super().close()
        close_scene()
