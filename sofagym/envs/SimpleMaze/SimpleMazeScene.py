# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Mar 23 2021"

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

import os
from SimpleMazeToolbox import rewardShaper, goalSetter
from splib3.animation import AnimationManagerController


path = os.path.dirname(os.path.abspath(__file__))
path_mesh = path + '/mesh/'


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=[0.0, 0.0, 0.0])
    return goal_mo


def createScene(root, config={"source": [0, 1000, 0],
                              "target": [0, 0, 0],
                              "goal_node": 0,
                              "goalPos": [0.0, 0.0, 0.0]}, mode='simu_and_visu'):

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

    # VISU ###################
    if visu:
        root.addObject('VisualStyle', displayFlags="showBehaviorModels showCollisionModels hideInteractionForceFields")

    if True:
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

    sphere = root.addChild("sphere")
    if True:
        sphere.addObject("EulerImplicitSolver", rayleighMass='5')
        sphere.addObject("SparseLDLSolver", name="ldl")
        sphere.addObject("GenericConstraintCorrection", solverName='ldl')
    ball_mo = sphere.addObject("MechanicalObject", name="sphere_mo", template='Vec3', position=[0.0, 6.0, 2.0])
    sphere.addObject("UniformMass", totalMass=1000)
   
    visu = object.addChild('Visu')
    visu.addObject('MeshOBJLoader', name='loader', filename=ball.obj, scale3d=1)
    visu.addObject('OglModel', src='@loader', color=[255, 0, 0, 255])
    visu.addObject('RigidMapping')


    object.addObject('GenerateRigidMass', name='mass', density=density, src=visu.loader.getLinkPath())
    object.mass.init()
    translation = list(object.mass.centerToOrigin.value)
    object.addObject('UniformMass', vertexMass="@mass.rigidMass")

    visu.loader.translation = translation

    if withCollision:
        collision = object.addChild('Collision')
        collision.addObject('MeshOBJLoader', name='loader', filename=collisionFilename, scale3d=scale)
        collision.addObject('MeshTopology', src='@loader')
        collision.addObject('MechanicalObject', translation=translation)
        collision.addObject('TriangleCollisionModel', group = collisionGroup)
        collision.addObject('LineCollisionModel', group = collisionGroup)
        collision.addObject('PointCollisionModel', group = collisionGroup)
        collision.addObject('RigidMapping')

    maze = model.addChild("maze")
    maze.addObject("MeshSTLLoader", name="loader", filename=path_mesh+"maze_4_coarse.stl",
                   translation=[-50.0, 0.0, 50.0], rotation=[-90.0, 0.0, 0.0])
    maze.addObject("Mesh", src='@loader')
    maze.addObject("MechanicalObject", name='maze_mesh_mo')
    if True:
        maze.addObject("Triangle", group=1)
        maze.addObject("Line", group=1)
        maze.addObject("Point", group=1)
        maze.addObject("RigidMapping")

    path_node = maze.addChild("Path")
    p_mesh = path_node.addObject('MeshObjLoader', filename=path_mesh+"path.obj", flipNormals=True, triangulate=True,
                                 name='meshLoader', translation=[-50.0, 0.0, 50.0])
    p_mo = path_node.addObject("MechanicalObject", template="Rigid3d", name="dofs", position="@meshLoader.position",
                               showObject=True, showObjectScale=1.0)
    if True:
        path_node.addObject("RigidRigidMapping", input=maze_mo.getLinkPath(), output=p_mo.getLinkPath())

    goal_mo = add_goal_node(root)

    root.addObject(rewardShaper(name="Reward", rootNode=root, goal_node=config['goalList'][config['goal_node']],
                                path_mesh=p_mesh, path_mo=p_mo, ball_mo=ball_mo))
    root.addObject(goalSetter(name="GoalSetter", rootNode=root, goalMO=goal_mo, goalPos=config['goalPos']))

    if visu:
        source = config["source"]
        target = config["target"]
        root.addObject("LightManager")
        root.addObject("SpotLight", position=source, direction=[0, -1, 0])
        root.addObject("InteractiveCamera", name="camera", position=source, lookAt=target, zFar=500)

    return root
