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
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from sofagym.header import addHeader as header
from sofagym.header import addVisu as visu

from BubbleMotion import BubbleMotion
from BubbleMotionToolbox import rewardShaper, sceneModerator, applyAction, goalSetter

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


def add_goal_node(root, pos):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=0.5,
                             showColor=[0, 1, 0, 0.5], position= pos)
    return goal_mo


def createScene(rootNode, config={"source": [5, -5, 20],
                                  "target": [5, 5, 0],
                                  "goalPos": [7, 0, 20],
                                  "seed": None,
                                  "zFar":4000,
                                  "dt": 0.01,
                                  "max_pressure": 40,
                                  "init_pos": [5, 5, 5],
                                  "board_dim": 8},
                mode='simu_and_visu'):

    header(rootNode, alarmDistance=0.3, contactDistance=0.1, tolerance=1e-4, maxIterations=100, gravity=[0, 0, -981.0],
           dt=config['dt'])

    position_spot = [[0, 0, -25]]
    direction_spot = [[0.0, 0, 1]]
    visu(rootNode, config, position_spot, direction_spot, cutoff=250)

    bd = config["board_dim"]
    pos_goal = [1+bd*np.random.random(), 1+bd*np.random.random(), 2]
    add_goal_node(rootNode, pos_goal)

    init_pos = config["init_pos"]
    bubblemotion_config = {'init_pos': init_pos, "dt": config["dt"], "max_pressure": config["max_pressure"]}

    bubblemotion = BubbleMotion(bubblemotion_config=bubblemotion_config)
    bubblemotion.onEnd(rootNode)

    rootNode.addObject(goalSetter(name="GoalSetter", goalPos=pos_goal))
    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode))
    rootNode.addObject(sceneModerator(name="sceneModerator", bubblemotion=bubblemotion))
    rootNode.addObject(applyAction(name="applyAction", root=rootNode, bubblemotion=bubblemotion))

    # rootNode.addObject(ControllerBubbleMotion(name="Controller", root= rootNode, bubblemotion = bubblemotion, max_pressure = config["max_pressure"]))

    # if VISUALISATION:
    #     print(">> Add runSofa visualisation")
    #     from common.visualisation import visualisationRunSofa, get_config
    #     path = str(pathlib.Path(__file__).parent.absolute())+"/../../../"
    #     config = get_config(path+"Results/config_abstractjimmy-v0.txt")
    #     config_env = config['env']
    #     actions = config['actions']
    #     scale = config_env['scale_factor']
    #     rootNode.addObject(visualisationRunSofa(name="visualisationRunSofa", root = rootNode, actions = actions, scale = scale) )
