# -*- coding: utf-8 -*-
"""Visualise the evolution of the scene in a runSofa way.

"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "May 4 2021"

import json
import Sofa


def get_config(path="./config.txt"):
    """Recover the config of the visualization in a json file.
    The config contains:
    - The name of the environment.
    - The environment configuration.
    - The action to perform in the environment.

    Parameters:
    ----------
        path: string, default ./config.txt
            The path to the file containing the configuration.

    Returns:
    -------
        The configuration.

    """
    with open(path, 'r') as outfile:
        config = json.load(outfile)
    return config


class visualisationRunSofa(Sofa.Core.Controller):
    """Controller to apply the action in the environment in runSofa.

    Note:
    -----
        This controller is designed for AbstractJimmy.

    Arguments:
    ----------
        root: the root of the scene.
        actions: the actions we want to apply in the environment.
        scale: scale factor, ie the nomber of time we apply an action.
        apply_action: controller that apply the action.
        current_idx: the action we consider.
        already_done: the nomber of time we already apply one action.
        current_incr: the increment that correspond to the current_action.

    Methods:
    -------
        __init__: classical init function.
    """
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root = kwargs["root"]
        self.actions = kwargs["actions"]
        self.scale = kwargs["scale"]

        self.apply_action = self.root.applyAction

        self.current_idx = 0
        self.already_done = 0
        self.current_incr = None

    def onAnimateBeginEvent(self, event):
        try:
            if self.already_done % self.scale == 0:
                current_action = self.actions[self.current_idx]
                # print(current_action)
                self.current_incr = self.apply_action.compute_action(current_action, self.scale)
                self.current_idx += 1

            self.apply_action.apply_action(self.current_incr)
            self.already_done += 1
        except:
            pass


"""HOW TO USE: in config: list of actions
    print(">> Add visualisationRunSofa")
    from visualisation import visualisationRunSofa, get_config
    config = get_config()
    config_env = config['env']
    actions = config['actions']
    scale = config_env['scale_factor']

    rootNode.addObject(visualisationRunSofa(name="visualisationRunSofa", root = rootNode, actions = actions, scale = scale))
"""
