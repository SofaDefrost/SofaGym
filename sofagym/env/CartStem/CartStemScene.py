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

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from common.header import addHeader as header
from common.header import addVisu as visu

from CartStem import CartStem
from CartStemToolbox import rewardShaper, sceneModerator, applyAction, goalSetter


def add_goal_node(root, pos):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=0.5,
                             showColor=[1, 0, 0, 0.5], position= pos)
    return goal_mo


def createScene(rootNode, config={"source": [0, -70, 10],
                                  "target": [0, 0, 10],
                                  "goalPos": [7, 0, 20],
                                  "seed": None,
                                  "zFar":4000,
                                  "init_x": 0,
                                  "max_move": 40,
                                  "dt": 0.01},
                mode='simu_and_visu'):

    header(rootNode, alarmDistance=1.0, contactDistance=0.1, tolerance = 1e-6, maxIterations=100,
           gravity=[0, 0, -981.0], dt=config['dt'])

    position_spot = [[0, -50, 10]]
    direction_spot = [[0.0, 1, 0]]
    visu(rootNode, config, position_spot, direction_spot, cutoff = 250)

    max_move = config['max_move']
    init_x = config["init_x"]

    cosserat_config = {'init_pos': [init_x, 0, 0], 'tot_length': 25, 'nbSectionS': 1, 'nbFramesF': 20}
    cartstem_config = {"init_pos": [init_x, 0, 0], "cart_size": [2, 2, 5], "max_move": max_move,  "max_v": 2,
                       "dt": config["dt"],  "cosserat_config": cosserat_config}

    cartstem = CartStem(cartstem_config=cartstem_config)
    cartstem.onEnd(rootNode)
    cartstem.cart.addObject('ConstantForceField', totalForce=[0, 0, 0, 0, 0, 0])

    rootNode.addObject(goalSetter(name="GoalSetter"))
    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, max_dist= cartstem_config['max_move']))
    rootNode.addObject(sceneModerator(name="sceneModerator",  cartstem = cartstem))
    rootNode.addObject(applyAction(name="applyAction", root= rootNode, cartstem=cartstem))

    # rootNode.addObject(ControllerCartStem(name="Controller", root= rootNode, cartstem = cartstem))

    # if VISUALISATION:
    #     print(">> Add runSofa visualisation")
    #     from common.visualisation import visualisationRunSofa, get_config
    #     path = str(pathlib.Path(__file__).parent.absolute())+"/../../../"
    #     config = get_config(path+"Results/config_abstractjimmy-v0.txt")
    #     config_env = config['env']
    #     actions = config['actions']
    #     scale = config_env['scale_factor']
    #     rootNode.addObject(visualisationRunSofa(name="visualisationRunSofa", root = rootNode, actions = actions, scale = scale) )
