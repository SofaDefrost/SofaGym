# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Mar 23 2021"

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from Maze import Maze
from Sphere import Sphere
from SimpleMazeToolbox import goalSetter, rewardShaper
from splib3.animation import AnimationManagerController

path = os.path.dirname(os.path.abspath(__file__))
path_mesh = path + '/mesh/'


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=[0.0, 0.0, 0.0])
    goal.addObject("RigidMapping", name='mapping', input=root.model.rigid_maze_mo.getLinkPath(), output=goal_mo.getLinkPath())
    
    return goal


def createScene(root, config={"source": [0, 1000, 0],
                              "target": [0, 0, 0],
                              "goal_node": 0,
                              "goalPos": 0,
                              "dt": 0.01}, mode='simu_and_visu'):

    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    root.addObject("RequiredPlugin", name="SoftRobots")
    root.addObject("RequiredPlugin", name="SofaPython3")
    root.addObject("RequiredPlugin", name="SofaOpenglVisual")
    root.addObject("RequiredPlugin", name="SofaMiscCollision")
    root.addObject("RequiredPlugin", name="SofaSparseSolver")

    root.gravity = [0.0, -981, 0.0]
    root.dt.value = config["dt"]

    # VISU ###################
    if visu:
        root.addObject('VisualStyle', displayFlags="showBehaviorModels showCollisionModels hideInteractionForceFields")

    root.addObject('DefaultPipeline')
    root.addObject('BruteForceDetection')
    root.addObject('RuleBasedContactManager', responseParams="mu=0.0001", name='Response', response='FrictionContactConstraint')
    root.addObject('LocalMinDistance', alarmDistance=6, contactDistance=0.1, angleCone=0.01)
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver', tolerance="1e-6", maxIterations="1000")

    root.addObject(AnimationManagerController(root, name="AnimationManager"))

    model = root.addChild('model')
    model.addObject('EulerImplicitSolver', firstOrder=True)
    model.addObject('CGLinearSolver', iterations=100, threshold=1e-5, tolerance=1e-5)
    maze_mo = model.addObject('MechanicalObject', name='rigid_maze_mo', template='Rigid3d',
                              position=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], showObject=True, showObjectScale=10)
    model.addObject('RestShapeSpringsForceField', angularStiffness=100, stiffness=100)
    model.addObject('UncoupledConstraintCorrection', compliance='1e-10  1e-10  0 0 1e-10  0 1e-10 ')

    maze = model.addChild(Maze())
    maze.addObject("RigidMapping")
    
    sphere = root.addChild(Sphere())
    ball_mo = sphere.sphere_mo

    path_node = maze.addChild("Path")
    p_mesh = path_node.addObject('MeshObjLoader', filename=path_mesh+"path.obj", flipNormals=True, triangulate=True,
                                 name='meshLoader', translation=[-50.0, 0.0, 50.0])
    p_mo = path_node.addObject("MechanicalObject", template="Vec3d", name="dofs", position="@meshLoader.position",
                               showObject=True, showObjectScale=1.0)
    path_node.addObject("RigidMapping", input=maze_mo.getLinkPath(), output=p_mo.getLinkPath())

    goal = add_goal_node(root)

    root.addObject(rewardShaper(name="Reward", rootNode=root, goal_node=config['goalPos'],
                                path_mesh=p_mesh, path_mo=p_mo, ball_mo=ball_mo))
    root.addObject(goalSetter(name="GoalSetter", rootNode=root, goal=goal, goalPos=config['goalPos']))

    if visu:
        source = config["source"]
        target = config["target"]
        root.addObject("LightManager")
        root.addObject("SpotLight", position=source, direction=[0, -1, 0])
        root.addObject("InteractiveCamera", name="camera", position=source, lookAt=target, zFar=500)

    return root
