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

from sofagym.header import addHeader as header
from sofagym.header import addVisu as visu

from CartStemContact import CartStem, Contacts
from CartStemContactToolbox import rewardShaper, goalSetter, sceneModerator, applyAction


def add_goal_node(root, pos):
    goal = root.addChild("Goal")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=False)
    goal.addObject('MeshOBJLoader', name="loader", filename='mesh/cylinder.obj', scale3d=[0.05, 3, 0.05],
                   rotation=[90, 0, 0], translation=[pos[0], pos[1], pos[2]-20])
    goal.addObject('OglModel', src='@loader', color=[1, 0, 0, 0.5])
    return goal_mo


def createScene(rootNode, config={"source": [0, -50, 10],
                                  "target": [0, 0, 10],
                                  "goalPos": [7, 0, 20],
                                  "seed": None,
                                  "zFar": 4000,
                                  "init_x": 0,
                                  "cube_x": [-6, 6],
                                  "max_move": 7.5,
                                  "dt": 0.01},
                         mode='simu_and_visu'):

    header(rootNode, alarmDistance=1.0, contactDistance=0.1, tolerance=1e-6, maxIterations=100, gravity=[0,0,-981.0],
           dt=config['dt'])

    position_spot = [[0, -50, 10]]
    direction_spot = [[0.0, 1, 0]]
    visu(rootNode, config, position_spot, direction_spot, cutoff=250)

    max_move = config['max_move']
    assert config['cube_x'][0] < config['cube_x'][1]
    bound = [config['cube_x'][0]+3, config['cube_x'][1]-3]
    init_x = max(-min(config["init_x"], bound[1]), bound[0])

    max_v = 2
    cosserat_config = {'init_pos': [init_x, 0, 0], 'tot_length': 25, 'nbSectionS': 1, 'nbFramesF': 20}
    cartstem_config = {"init_pos": [init_x, 0, 0], "cart_size": [2, 2, 5], "max_move": max_move,  "max_v": max_v,
                       "dt": config["dt"],  "cosserat_config": cosserat_config}
    contact_config = {"init_pos": [0, 0, 12], "cube_size": [2, 1, 2], "cube_x": config["cube_x"]}

    cartstem = CartStem(cartstem_config=cartstem_config)
    cartstem.onEnd(rootNode)

    contacts = Contacts(contact_config=contact_config)
    contacts.onEnd(rootNode)

    add_goal_node(rootNode, config["goalPos"])

    rootNode.addObject(goalSetter(name="GoalSetter", goalPos=config["goalPos"]))
    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, max_dist=cartstem_config['max_move']))
    rootNode.addObject(sceneModerator(name="sceneModerator",  cartstem=cartstem, contacts=contacts))
    rootNode.addObject(applyAction(name="applyAction", root=rootNode, cartstem=cartstem))

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
