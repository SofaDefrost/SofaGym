# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 01 2021"

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

mesh_path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'

import numpy as np
from Maze import Maze
from MazeToolbox import goalSetter, rewardShaper
from Sphere import Sphere
from splib3.animation import animate
from splib3.animation import AnimationManagerController
from splib3.numerics import RigidDof
from stlib3.scene import ContactHeader, Scene
from tripod import Tripod


def setupanimation(actuators, step, angularstep, factor):
    """This function is called repeatidely in an animation.
       It moves the actuators by translating & rotating them according to the factor
       value.
    """
    for actuator in actuators:
        rigid = RigidDof(actuator.ServoMotor.ServoBody.dofs)
        rigid.setPosition(rigid.rest_position + rigid.forward * step * factor)
        actuator.angleIn = angularstep * factor


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=[0.0, 0.0, 0.0])
    goal.addObject("RigidMapping", name='mapping', input=root.Modelling.Tripod.RigidifiedStructure.FreeCenter.dofs.getLinkPath(), output=goal_mo.getLinkPath())

    return goal


def createScene(rootNode, config={"source": [0, 300, 0],
                                  "target": [0, 0, 0],
                                  "goalList": [0, 0, 0],
                                  "goal_node": 0,
                                  "goalPos": 0,
                                  "dt": 0.01}, mode='simu_and_visu'):
    
    pluginList = ["ArticulatedSystemPlugin",
                  "Sofa.Component.AnimationLoop",
                  "Sofa.Component.Collision.Detection.Algorithm",
                  "Sofa.Component.Collision.Detection.Intersection",
                  "Sofa.Component.Collision.Geometry",
                  "Sofa.Component.Collision.Response.Contact",
                  "Sofa.Component.Constraint.Lagrangian.Correction",
                  "Sofa.Component.Constraint.Lagrangian.Solver",
                  "Sofa.Component.Constraint.Projective",
                  "Sofa.Component.Engine.Select",
                  "Sofa.Component.IO.Mesh",
                  "Sofa.Component.LinearSolver.Direct",
                  "Sofa.Component.Mapping.MappedMatrix",
                  "Sofa.Component.Mass",
                  "Sofa.Component.SolidMechanics.FEM.Elastic",
                  "Sofa.Component.SolidMechanics.Spring",
                  "Sofa.Component.StateContainer",
                  "Sofa.Component.Topology.Container.Constant",
                  "Sofa.Component.Topology.Container.Dynamic",
                  "Sofa.Component.Visual",
                  "Sofa.GL.Component.Rendering3D",
                  "Sofa.GUI.Component",
                  'SoftRobots',
                  'SofaPreconditioner',
                  'SofaPython3',
                  'SofaOpenglVisual',
                  'SofaMiscCollision',
                  'SofaBoundaryCondition',
                  'SofaConstraint',
                  'SofaDeformable',
                  'SofaEngine',
                  'SofaGeneralAnimationLoop',
                  'SofaGeneralEngine',
                  'SofaGeneralRigid',
                  'SofaImplicitOdeSolver',
                  'SofaLoader',
                  'SofaMeshCollision',
                  'SofaMiscMapping',
                  'SofaRigid',
                  'SofaSimpleFem',
                  'Sofa.Component.Mapping.Linear',
                  'Sofa.Component.Mapping.NonLinear',
                  'Sofa.GL.Component.Shader']

    # Choose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    scene = Scene(rootNode, gravity=[0.0, -9810, 0.0], dt=config["dt"], plugins=pluginList, iterative=False)
    scene.addMainHeader()
    scene.addObject('DefaultVisualManagerLoop')
    scene.addObject('FreeMotionAnimationLoop')
    scene.addObject('GenericConstraintSolver', name='solver', maxIterations=1000, tolerance=1e-6)
    scene.Simulation.addObject('GenericConstraintCorrection')
    scene.Settings.mouseButton.stiffness = 10
    scene.Simulation.TimeIntegrationSchema.rayleighStiffness = 0.05
    ContactHeader(rootNode, alarmDistance=0.5, contactDistance=0.2, frictionCoef=0.2)

    rootNode.addObject(AnimationManagerController(rootNode, name="AnimationManager"))
    
    # Visu
    if visu:
        scene.VisualStyle.displayFlags = "showForceFields showBehavior showCollisionModels showVisualModels"

    # Tripod
    tripod = scene.Modelling.addChild(Tripod())
    tripod.addCollision()
    scene.Simulation.addChild(tripod)

    # Maze
    maze = tripod.RigidifiedStructure.FreeCenter.addChild(Maze())
    maze.addObject("RigidMapping", index=0)

    # Sphere
    sphere = scene.Simulation.addChild(Sphere(withSolver=False))
    ball_mo = sphere.sphere_mo

    path = maze.addChild("Path")
    p_mesh = path.addObject('MeshObjLoader', filename=mesh_path+"path.obj", flipNormals=True, triangulate=True,
                            name='meshLoader', translation=[-50, 0, 50])
    p_mo = path.addObject("MechanicalObject", template="Vec3d", name="dofs", position="@meshLoader.position",
                          showObject=True, showObjectScale=1.0)
    path.addObject("RigidMapping", input=tripod.RigidifiedStructure.RigidParts.dofs.getLinkPath(), output='@./',
                   index=3)

    goal = add_goal_node(rootNode)

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goal_node=config['goalPos'],
                                    path_mesh=p_mesh, path_mo=p_mo, ball_mo=ball_mo))
    rootNode.addObject(goalSetter(name="GoalSetter", rootNode=rootNode, goal=goal, goalPos=config['goalPos']))
    
    if visu:
        source = config["source"]
        target = config["target"]
        rootNode.addObject("LightManager")
        spotloc = [0, source[1], 0]
        rootNode.addObject("SpotLight", position=spotloc, direction=[0, -np.sign(source[1]), 0])
        rootNode.addObject("InteractiveCamera", name="camera", position=source, orientation=[-0.414607,-0.196702,-0.0234426,0.888178], lookAt=target, zFar=500)

    actuators=[tripod.ActuatedArm0, tripod.ActuatedArm1, tripod.ActuatedArm2]
    animate(setupanimation, {"actuators": actuators, "step": 35.0, "angularstep": -1.4965}, duration=0.2)

    return rootNode
