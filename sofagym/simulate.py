# -*- coding: utf-8 -*-
"""Tools to create the scene, run one step and recover the OpenGL context from
Sofa.
"""

__authors__ = ("PSC", "emenager")
__contact__ = ("pierre.schegg@robocath.com", "etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Robocath, Inria"
__date__ = "Oct 7 2020"

import Sofa
import SofaRuntime

import importlib


def init_simulation(config, _startCmd=None, mode="simu_and_visu"):
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
    scene = config['scene']
    # Import the create_scene corresponding to the environment

    try:
        create_scene = importlib.import_module("sofagym.envs."+scene+"." + scene + "Scene").createScene
    except:
        print("sofagym.envs."+scene+"." + scene + "Scene")
        raise NotImplementedError("Importing your SOFA Scene Failed")

    # Load the scene
    root = Sofa.Core.Node("root")
    SofaRuntime.importPlugin("Sofa.Component")
    create_scene(root,  config, mode = mode)
    Sofa.Simulation.init(root)

    # Realise action from history
    if config['start_from_history'] is not None and _startCmd is not None:
        print(">>   Start from history ...")
        render = config['render']
        config.update({'render': 0})

        for action in config['start_from_history']:
            step_simulation(root, config, action, _startCmd, None)

        config.update({'render': render})
        print(">>   ... Done.")

    # Init Reward and GoalSetter
    if config["goal"]:
        root.GoalSetter.update(config['goalPos'])
    
    root.Reward.update(0)

    try:
        root.StateInitializer.init_state(config["init_states"])
    except AttributeError as error:
        print(error)

    if 'time_before_start' in config:
        print(">>   Time before start:", config["time_before_start"], "steps. Initialization ...")
        for i in range(config["time_before_start"]):
            Sofa.Simulation.animate(root, config["dt"])
        print(">>   ... Done.")
        
        # Update Reward and GoalSetter
        if config["goal"]:
            root.GoalSetter.update(config['goalPos'])
            root.Reward.update(config['goalPos'])
        else:
            root.Reward.update()

    return root


def step_simulation(root, config, action, _startCmd, _getPos, viewer=None):
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
    if config["goal"]:
        goal = config['goalPos']
        root.GoalSetter.set_mo_pos(goal)
    
    render = config['render']
    surface_size = config['display_size']


    # Create the command from action
    _startCmd(root, action, config["dt"]*(config["scale_factor"]-1))
    pos = []
    # Realise scale_factor simulation steps of 0.01 s
    for i in range(config["scale_factor"]):
        Sofa.Simulation.animate(root, config["dt"])
        if render == 2:
            pos.append(_getPos(root))
            if viewer is not None:
                viewer.render_simulation(root)

    if render == 1:
        pos.append(_getPos(root))

    return pos
