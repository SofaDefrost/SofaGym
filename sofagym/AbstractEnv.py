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

from typing import Optional

import numpy as np
import copy
import os

import splib3

from sofagym.rpc_server import start_server, add_new_step, get_result, clean_registry, close_scene
from sofagym.viewer import Viewer

import importlib

import Sofa
import SofaRuntime


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
                - dt: float that define time step.
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
    def __init__(self, default_config, config=None, render_mode: Optional[str]=None, root=None):
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
        self.config = copy.deepcopy(default_config)
        self.config["dt"] = self.config.get('dt', 0.01)
        if config is not None:
            self.config.update(config)

        self.scene = self.config['scene']

        self._getState = importlib.import_module("sofagym.envs."+self.scene+"."+self.scene+"Toolbox").getState
        self._getReward = importlib.import_module("sofagym.envs."+self.scene+"."+self.scene+"Toolbox").getReward
        self._startCmd = importlib.import_module("sofagym.envs."+self.scene+"."+self.scene+"Toolbox").startCmd
        self._getPos = importlib.import_module("sofagym.envs."+self.scene+"."+self.scene+"Toolbox").getPos
        
        try:
            self.create_scene = importlib.import_module("sofagym.envs."+self.scene+"." + self.scene + "Scene").createScene
        except Exception as exc:
            print("sofagym.envs."+self.scene+"." + self.scene + "Scene")
            raise NotImplementedError("Importing your SOFA Scene Failed") from exc

        self.viewer = None
        self.render_mode = render_mode

        self.past_actions = []

        self.pos = []
        self.past_pos = []

        self.num_envs = 40

        self.np_random = None

        self.seed(self.config['seed'])

        self.timer = 0
        self.timeout = self.config["timeout"]

        self.init_save_paths()

        self.goal = None
        if self.config["goal"]:
            self.goalList = self.config["goalList"]
            self.init_goal()

        self.root = root

    def init_goal(self):
        # Set a new random goal from the list
        id_goal = self.np_random.choice(range(len(self.goalList)))
        self.config.update({'goal_node': id_goal})
        self.goal = self.goalList[id_goal]
        self.config.update({'goalPos': self.goal})

    def init_save_paths(self):
        """Create directories to save results and images.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.
        """
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

    def init_root(self):
        self.init_simulation()

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

        self.pos = self.step_simulation(action)

        self.past_actions.append(action)
        self.past_pos.append(self.pos)

        obs = np.array(self._getState(self.root), dtype=np.float32)
        done, reward = self._getReward(self.root)

        # Avoid long explorations by using a timer.
        self.timer += 1
        if self.timer >= self.config["timer_limit"]:
            # reward = -150
            truncated = True
        
        info = {} #(not use here)

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
        self.viewer = None

        splib3.animation.animate.manager = None

        if self.config["goal"]:
            self.init_goal()

        self.timer = 0
        self.past_actions = []
        self.pos = []
        self.past_pos = []

        Sofa.Simulation.reset(self.root)
        self.root = None
        self.init_simulation()
        
        obs = np.array(self._getState(self.root), dtype=np.float32)
        
        return obs

    def render(self, mode):
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
        self.render_mode = mode

        # Define the viewer at the first run of render.
        if self.viewer is None:
            display_size = self.config["display_size"]  # Sim display
            if 'zFar' in self.config:
                zFar = self.config['zFar']
            else:
                zFar = 0
            self.viewer = Viewer(self, self.root, display_size, zFar=zFar, save_path=self.config["save_path_image"])
        # Use the viewer to display the environment.
        return self.viewer.render()

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

    def init_simulation(self, mode='simu_and_visu'):
        """Function to create scene and initialize all variables.

        Parameters:
        ----------
            config: Dictionary.
                Configuration of the environment.
            _startCmd: function
                Initialize the command.
            mode: string, default = 'simu_and_visu'
                Init a scene with or without visu and computations.
                In ['simu', 'visu', 'simu_and_visu']

        Returns:
        -------
            root: <Sofa.Core>
                The loaded and initialized scene.

        """
        # Load the scene
        self.root = Sofa.Core.Node("root")

        SofaRuntime.importPlugin("Sofa.Component")
        self.create_scene(self.root, self.config, mode=mode)
        Sofa.Simulation.init(self.root)

        # Realise action from history
        if self.config['start_from_history'] is not None and self._startCmd is not None:
            print(">>   Start from history ...")
            render = self.config['render']
            self.config.update({'render': 0})

            for action in self.config['start_from_history']:
                self.step_simulation(action)

            self.config.update({'render': render})
            print(">>   ... Done.")

        # Init Reward and GoalSetter
        if self.config["goal"]:
            self.root.GoalSetter.update(self.goal)
        
        self.root.Reward.update(0)

        if self.config["randomize_states"]:
            self.root.StateInitializer.init_state(self.config["init_states"])

        if 'time_before_start' in self.config:
            print(">>   Time before start:", self.config["time_before_start"], "steps. Initialization ...")
            for _ in range(self.config["time_before_start"]):
                Sofa.Simulation.animate(self.root, self.config["dt"])
            print(">>   ... Done.")
            
            # Update Reward and GoalSetter
            if self.config["goal"]:
                self.root.GoalSetter.update(self.goal)
            
            self.root.Reward.update(self.goal)

    def step_simulation(self, action):
        """Realise one step in the simulation.

        Apply action and execute 5 times scale_factor simulation steps of dt s.

        Parameters:
        ----------
            root: <Sofa.Core>
                The scene.
            config: Dictionary
                The configuration of the environment.
            action: int
                The action to apply in the environment.
            _startCmd: function
                Initialize the command.
            _getPos: function
                Get the position of the object in the scene.

        Returns:
        --------
            position(s): list
                The positions of object(s) in the scene.

        """
        if self.config["goal"]:
            goal = self.config['goalPos']
            self.root.GoalSetter.set_mo_pos(goal)
        
        render = self.config['render']
        surface_size = self.config['display_size']

        # Create the command from action
        self._startCmd(self.root, action, self.config["dt"]*(self.config["scale_factor"]-1))
        pos = []
        # Realise scale_factor simulation steps of 0.01 s
        for _ in range(self.config["scale_factor"]):
            Sofa.Simulation.animate(self.root, self.config["dt"])
            
            #if render == 2:
            #    pos.append(self._getPos(self.root))
            #    if self.viewer is not None:
            #        self.viewer.render_simulation(self.root)

        if render == 1:
            pos.append(self._getPos(self.root))

        return pos




class ServerEnv(AbstractEnv):
    def __init__(self, default_config, config=None, render_mode: Optional[str]=None, root=None):
        super().__init__(default_config, config, root=root)
        
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
    
    def reset(self):
        """Reset simulation.

        Parameters:
        ----------
            None.

        Returns:
        -------
            obs, info
        """
        self.clean()
        self.viewer = None

        splib3.animation.animate.manager = None
        
        if self.config["goal"]:
            self.init_goal()

        self.timer = 0
        self.past_actions = []
        
        return
    
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
