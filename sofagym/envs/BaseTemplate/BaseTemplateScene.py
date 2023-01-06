import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from splib3.animation import AnimationManagerController
from math import cos, sin
import numpy as np
from splib3.objectmodel import SofaPrefab, SofaObject
from splib3.numerics import Vec3, Quat


from sofagym.envs.BaseTemplate.BaseTemplateToolbox import rewardShaper, goalSetter

import os
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=[0.0, -100.0, 100.0])
    return goal_mo


def effectorTarget(parentNode, position=[0., 0., 200]):
    target = parentNode.addChild("Target")
    target.addObject("EulerImplicitSolver", firstOrder=True)
    target.addObject("CGLinearSolver")
    target.addObject("MechanicalObject", name="dofs", position=position, showObject=True, showObjectScale=3,
                     drawMode=2, showColor=[1., 1., 1., 1.])
    target.addObject("UncoupledConstraintCorrection")
    return target


def createScene(rootNode, config={"source": [-600.0, -25, 100],
                                  "target": [30, -25, 100],
                                  "goalPos": [0, 0, 0]}, mode='simu_and_visu'):

    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    rootNode.addObject("RequiredPlugin", name="SoftRobots")
    rootNode.addObject("RequiredPlugin", name="SofaSparseSolver")
    rootNode.addObject("RequiredPlugin", name="SofaPreconditioner")
    rootNode.addObject("RequiredPlugin", name="SofaPython3")
    rootNode.addObject('RequiredPlugin', name='BeamAdapter')
    rootNode.addObject('RequiredPlugin', name='SofaOpenglVisual')
    rootNode.addObject('RequiredPlugin', name="SofaMiscCollision")
    rootNode.addObject("RequiredPlugin", name="SofaBoundaryCondition")
    rootNode.addObject("RequiredPlugin", name="SofaConstraint")
    rootNode.addObject("RequiredPlugin", name="SofaEngine")
    rootNode.addObject('RequiredPlugin', name='SofaImplicitOdeSolver')
    rootNode.addObject('RequiredPlugin', name='SofaLoader')
    rootNode.addObject('RequiredPlugin', name="SofaSimpleFem")

    if visu:
        source = config["source"]
        target = config["target"]
        rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                       'hideMappings hideForceFields showWireframe')
        rootNode.addObject("LightManager")

        spotLoc = [2*source[0], 0, 0]
        rootNode.addObject("SpotLight", position=spotLoc, direction=[-np.sign(source[0]), 0.0, 0.0])
        rootNode.addObject("InteractiveCamera", name='camera', position=source, lookAt=target, zFar=500)
        rootNode.addObject('BackgroundSetting', color=[1, 1, 1, 1])
    if simu:
        rootNode.addObject('DefaultPipeline')
        rootNode.addObject('FreeMotionAnimationLoop')
        rootNode.addObject('GenericConstraintSolver', tolerance="1e-6", maxIterations="1000")
        rootNode.addObject('BruteForceDetection')
        rootNode.addObject('RuleBasedContactManager', responseParams="mu="+str(0.3), name='Response',
                           response='FrictionContactConstraint')
        rootNode.addObject('LocalMinDistance', alarmDistance=10, contactDistance=5, angleCone=0.01)

        rootNode.addObject(AnimationManagerController(name="AnimationManager"))

        rootNode.gravity.value = [0., -9810., 0.]

    rootNode.dt.value = 0.01

    simulation = rootNode.addChild("Simulation")

    if simu:
        simulation.addObject('EulerImplicitSolver', name='odesolver', firstOrder="0", rayleighMass="0.1",
                             rayleighStiffness="0.1")
        simulation.addObject('ShewchukPCGLinearSolver', name='linearSolver', iterations='500', tolerance='1.0e-18',
                             preconditioners="precond")
        simulation.addObject('SparseLDLSolver', name='precond')
        simulation.addObject('GenericConstraintCorrection', solverName="precond")

    trunk = Trunk(simulation, inverseMode=False)
    rootNode.trunk = trunk

    if visu:
        trunk.addVisualModel(color=[1., 1., 1., 0.8])
    trunk.fixExtremity()

    goal_mo = add_goal_node(rootNode)

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goalPos=config['goalPos']))
    rootNode.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalPos=config['goalPos']))

    return rootNode
