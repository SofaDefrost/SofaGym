# -*- coding: utf-8 -*-
"""AbstractEnv to make the link between Gym and Sofa.
"""

__authors__ = ("PSC", "dmarchal", "emenager")
__contact__ = ("pierre.schegg@robocath.com", "damien.marchal@univ-lille.fr", "etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Robocath, CNRS, Inria"
__date__ = "Oct 7 2020"

import gym
from gym.utils import seeding

import numpy as np
import copy
import os

import splib3

from sofagym.viewer import Viewer
from sofagym.rpc_server import start_server, add_new_step, get_result, clean_registry, close_scene


class AbstractEnv(gym.Env):
    """Use Sofa scene with a Gym interface.

    Methods:
    -------
        __init__: classical __init__ method.
        initialization: Initialization of all arguments.
        seed: Initialization of the seed.
        step: Realise a step in the environment.
        async_step: Realise a step without blocking queue.
        reset: Reset the environment and useful arguments.
        render: Use viewer to see the environment.
        _automatic_rendering: Automatically render the intermediate frames
            while an action is still ongoing.
        close: Terminate the simulation.
        configure: Add element in the configuration.
        clean: clean the registery.
        _formataction.. : transforme the type of action to use server.

    Arguments:
    ---------
        config: Dictionary.
            Contains the configuration of the environment.
            Minimum:
                - scene : the name of the simulation.
                    Note: define the name of the toolbox <scene>Toolbox and the
                    scene <scene>Scene in the directory ../<scene>.
                - deterministic: whether or not the environment is deterministic.
                - source,target: definition of the Sofa camera point of view.
                - goalList : list of the goals to reach (position or index).
                - start_node: the start node (position or index).
                - scale_factor: int that define the number of step in simulation.
                - timer_limit: int that define the maximum number of steps.
                - timeout: int that define the timeout for the server/client requests.
                - display_size: tuple of int that define the size of the Viewer
                    window.
                - save_path: path to save the image of the simulation.
                - render: wheter or not the viewer displays images.
                    0: no rendering.
                    1: render after simulation.
                    2: render all steps.
                    Warning: we can't change this value after initialization.
                - save_data: wheter or not the data are saved.
                - save_image: wheter or not the images are saved.
                - planning: if realise planning or not.
                - discrete: if the environment is discrete or not.
                - timer_limit: the limit of the time.
                - seed : the seed.
                - start_from_history: list of actions that have to be carried
                    out before starting the training.
                - python_version: the version of python.
                - time_before_start: initialize the simulation with time_before_start steps.
        observation_space: spaces.Box
            Define the size of the environment.
        past_actions: list of int.
            Keeps track of past actions. Allows you to retrieve past
            configurations of the environment.
        goalList: list
            List of possible objectives to be achieved.
        goal: list
            Current objective.
        num_envs: int
            The number of environment.
        np_random:  np.random.RandomState()
             Exposes a number of methods for generating random numbers
        viewer: <class viewer>
            Allows to manage the visual feedback of the simulation.
        automatic_rendering_callback:
            Callback function used in _automatic_rendering.
        timer:
            Number of steps already completed.
        deterministic:
            Whether the environment is deterministic or not.
        timeout:
            Number of times the queue is blocking. Allows to avoid blocking
            situations.

    Notes:
    -----
        It is necessary to define the specificity of the environment in a
        subclass.

    Usage:
    -----
        Use the reset method before launch the environment.


    """
    def __init__(self, config=None):
        """
        Classic initialization of a class in python.

        Parameters:
        ----------
        config: Dictionary or None, default = None
            Customisable configuration element.

        Returns:
        ---------
            None.

        """

        # Define a DEFAULT_CONFIG in sub-class.
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        self.initialization()

    def initialization(self):
        """Initialization of all parameters.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.
        """

        self.goalList = None
        self.goal = None
        self.past_actions = []

        
        self.num_envs = 40

        self.np_random = None

        self.seed(self.config['seed'])

        self.viewer = None
        self.automatic_rendering_callback = None
        

        self.timer = 0
        self.timeout = self.config["timeout"]

        # Start the server which distributes the calculations to its clients
        start_server(self.config)

        if 'save_data' in self.config and self.config['save_data']:
            save_path_results = self.config['save_path']+"/data"
            os.makedirs(save_path_results, exist_ok=True)
        else:
            save_path_results = None

        if 'save_image' in self.config and self.config['save_image']:
            save_path_image = self.config['save_path']+"/img"
            os.makedirs(save_path_image, exist_ok=True)
        else:
            save_path_image = None

        self.configure({"save_path_image": save_path_image, "save_path_results": save_path_results})

    def seed(self, seed=None):
        """
        Computes the random generators of the environment.

        Parameters:
        ----------
        seed: int, 1D array or None, default = None
            seed for the RandomState.

        Returns:
        ---------
            [seed]

        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _formataction(self, action):
        """Change the type of action to be in [list, float, int].

        Parameters:
        ----------
            action:
                The action with no control on the type.

        Returns:
        -------
            action: in [list, float, int]
                The action with  control on the type.
        """
        if isinstance(action, np.ndarray):
            action = action.tolist()
        elif isinstance(action, np.int64):
            action = int(action)
        elif isinstance(action, np.float64):
            action = float(action)
        elif isinstance(action, tuple):
             action = self._formatactionTuple(action)
        elif isinstance(action, dict):
            action = self._formatactionDict(action)
        return action

    def _formatactionTuple(self, action):
        """Change the type of tuple action to be in [list, float, int].

        Parameters:
        ----------
            action:
                The action with no control on the type.

        Returns:
        -------
            action:
                The action with  control on the type.
        """
        return self._formataction(action[0]), self._formataction(action[1])

    def _formatactionDict(self, action):
        """Change the type of tuple action to be in [list, float, int].

        Parameters:
        ----------
            action:
                The action with no control on the type.

        Returns:
        -------
            action:
                The action with  control on the type.
        """
        for key in action.keys():
            action[key] = self._formataction(action[key])

        return action

    def clean(self):
        """Function to clean the registery .

        Close clients who are processing unused sequences of actions (for
        planning)

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.
        """

        clean_registry(self.past_actions)

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
            done(bool):
                Whether the agent reaches the terminal state
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

        obs = np.array(results["observation"])  # to work with baseline
        reward = results["reward"]
        done = results["done"]

        # Avoid long explorations by using a timer.
        self.timer += 1
        if self.timer >= self.config["timer_limit"]:
            # reward = -150
            truncated = True
        info={}#(not use here)

        if self.config["planning"]:
            self.clean()
        return obs, reward, done, info

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
                return obs, reward, done, {}

        return LateResult(copy.copy(result_id))

    def reset(self):
        """Reset simulation.

        Parameters:
        ----------
            None.

        Returns:
        -------
            obs, info

        """
        self.close()
        self.initialization()

        splib3.animation.animate.manager = None
        if not self.goalList:
            self.goalList = self.config["goalList"]

        # Set a new random goal from the list
        id_goal = self.np_random.choice(range(len(self.goalList)))
        self.config.update({'goal_node': id_goal})
        self.goal = self.goalList[id_goal]

        self.timer = 0
        self.past_actions = []
        
        return 

    def render(self, mode='rgb_array'):
        """See the current state of the environment.

        Get the OpenGL Context to render an image (snapshot) of the simulation
        state.

        Parameters:
        ----------
            mode: string, default = 'rgb_array'
                Type of representation.

        Returns:
        -------
            None.
        """
        if self.config['render'] != 0:
            # Define the viewer at the first run of render.
            if not self.viewer:
                display_size = self.config["display_size"]  # Sim display
                if 'zFar' in self.config:
                    zFar = self.config['zFar']
                else:
                    zFar = 0
                self.viewer = Viewer(self, display_size, zFar=zFar, save_path=self.config["save_path_image"])

            # Use the viewer to display the environment.
            self.viewer.render()
        else:
            print(">> No rendering")

    def _automatic_rendering(self):
        """Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps
        corresponding to agent decision-making.
        If a callback has been set, use it to perform the rendering. This is
        useful for the environment wrappers such as video-recording monitor that
        need to access these intermediate renderings.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.

        """
        if self.viewer is not None:
            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render()

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
        if self.viewer is not None:
            self.viewer.close()

        close_scene()
        print("All clients are closed. Bye Bye.")

    def configure(self, config):
        """Update the configuration.

        Parameters:
        ----------
            config: Dictionary.
                Elements to be added in the configuration.

        Returns:
        -------
            None.

        """
        self.config.update(config)
