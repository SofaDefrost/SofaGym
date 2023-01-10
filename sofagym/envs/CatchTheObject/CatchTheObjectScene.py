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

import numpy as np
from CatchTheObject import Cart, Ball, Gripper
from CatchTheObjectToolbox import rewardShaper, sceneModerator, applyAction, goalSetter


def createScene(rootNode, config={"source": [0, -70, 10],
                                  "target": [0, 0, 10],
                                  "goalPos": [7, 0, 20],
                                  "seed": None,
                                  "zFar":4000,
                                  "dt": 0.01,
                                  "max_move": 10,
                                  "max_pressure": 15},
                mode='simu_and_visu'):

    header(rootNode, alarmDistance=3.0, contactDistance=0.1, tolerance=1e-6, maxIterations=100, gravity=[0, 0, -981.0],
           dt=config['dt'])

    position_spot = [[0, -50, 10]]
    direction_spot = [[0.0, 1, 0]]
    visu(rootNode, config, position_spot, direction_spot, cutoff=250)

    max_move = config['max_move']
    max_pressure = config['max_pressure']

    cart_config = {"init_pos": [0, 0, 0], "max_move": max_move, "init_x": -max_move + 2*max_move*np.random.random()}
    cart = Cart(cart_config=cart_config)
    cart.onEnd(rootNode)

    init_z = 20 + 7*np.random.random()
    ball_config = {"max_high": 27, "init_pos": [0, 0, init_z], "size_ball": 1, "mass_ball": 1}
    ball = Ball(ball_config=ball_config)
    ball.onEnd(rootNode)

    gripper_config = {"max_pressure": max_pressure}
    gripper = Gripper(gripper_config=gripper_config)
    gripper.onEnd(rootNode)

    rootNode.addObject(goalSetter(name="GoalSetter"))
    rootNode.addObject(sceneModerator(name="sceneModerator",  cart=cart, ball=ball, gripper=gripper))
    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode))
    rootNode.addObject(applyAction(name="applyAction", root=rootNode, cart=cart, gripper=gripper))

    # rootNode.addObject(ControllerCatchTheObject(name="Controller", root= rootNode, cart = cart, gripper = gripper, ball = ball))

    # if VISUALISATION:
    #     print(">> Add runSofa visualisation")
    #     from common.visualisation import visualisationRunSofa, get_config
    #     path = str(pathlib.Path(__file__).parent.absolute())+"/../../../"
    #     config = get_config(path+"Results/config_abstractjimmy-v0.txt")
    #     config_env = config['env']
    #     actions = config['actions']
    #     scale = config_env['scale_factor']
    #     rootNode.addObject(visualisationRunSofa(name="visualisationRunSofa", root = rootNode, actions = actions, scale = scale) )
