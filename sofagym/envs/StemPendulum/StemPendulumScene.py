# -*- coding: utf-8 -*-
"""Create the scene with the Abstraction of Jimmy.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

VISUALISATION = False

import sys
import importlib
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from sofagym.header import addHeader as header
from sofagym.header import addVisu as visu
from sofagym.utils import addRigidObject

from StemPendulum import StemPendulum
from StemPendulumToolbox import rewardShaper, sceneModerator, applyAction, goalSetter
from Controller import ControllerStemPendulum

def add_goal_node(root, pos):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=0.5,
                             showColor=[1, 0, 0, 0.5], position= pos)
    return goal_mo


def createScene(rootNode, config = {"source": [0, 0, 30],
                                    "target": [0, 0, 0],
                                    "goalPos": [7, 0, 20],
                                    "seed": None,
                                    "zFar":4000,
                                    "max_torque": 500,
                                    "dt": 0.01},
                         mode = 'simu_and_visu'):

    header(rootNode, alarmDistance=1.0, contactDistance=0.1, tolerance = 1e-6, maxIterations=100, gravity = [0,-981.0,0], dt = config['dt'])

    position_spot = [[0, 0, -25]]
    direction_spot = [[0.0, 0, 1]]
    visu(rootNode, config, position_spot, direction_spot, cutoff = 250)

    init_or = 360*np.random.random()
    beam_config = {'init_pos': [0, 0, 0], 'tot_length': 10, 'nbSections': 10}
    stempendulum_config = {"init_or": init_or, "base_size":  [0.05, 0.05, 0.05], "dt": config["dt"],
                            "beam_config":beam_config, "max_torque": config["max_torque"]}

    stempendulum = StemPendulum(stempendulum_config = stempendulum_config)
    stempendulum.onEnd(rootNode)

    rootNode.addObject(goalSetter(name="GoalSetter"))
    rootNode.addObject(sceneModerator(name="sceneModerator",  stempendulum = stempendulum))
    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode))
    rootNode.addObject(applyAction(name="applyAction", root= rootNode, stempendulum=stempendulum))

    # rootNode.addObject(ControllerStemPendulum(name="Controller", root= rootNode, stempendulum = stempendulum, max_torque = config['max_torque']))

    # if VISUALISATION:
    #     print(">> Add runSofa visualisation")
    #     from common.visualisation import visualisationRunSofa, get_config
    #     path = str(pathlib.Path(__file__).parent.absolute())+"/../../../"
    #     config = get_config(path+"Results/config_abstractjimmy-v0.txt")
    #     config_env = config['env']
    #     actions = config['actions']
    #     scale = config_env['scale_factor']
    #     rootNode.addObject(visualisationRunSofa(name="visualisationRunSofa", root = rootNode, actions = actions, scale = scale) )
