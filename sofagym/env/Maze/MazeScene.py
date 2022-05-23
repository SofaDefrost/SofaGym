# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 01 2021"

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from stlib3.stlib.physics.deformable import ElasticMaterialObject
from stlib3.splib.objectmodel import SofaPrefab, SofaObject
from stlib3.splib.numerics import *
from stlib3.stlib.components import addOrientedBoxRoi
from stlib3.stlib.physics.collision import CollisionMesh

from actuatedarm import ActuatedArm
from rigidification import Rigidify
from utils import Scene, setData
from MazeToolbox import rewardShaper, goalSetter

import math
import os
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
path_mesh = path + '/mesh/'


def to_euler(angle):

    return angle * 180/math.pi


@SofaPrefab
class Tripod(SofaObject):
    def __init__(self, parent, name="Tripod", radius=66, numMotors=3, angleShift=180.0, translation=[0.0, 0.0, 0.0],
                 rotation=[0.0, 0.0, 0.0]):
        self.node = parent.addChild(name)
        self.node.addChild("ElasticBody")
        ElasticMaterialObject(self.node.ElasticBody, name="ElasticMaterialObject", translation=translation,
                              rotation=rotation, volumeMeshFileName=path_mesh + "tripod_coarse_04_test.vtk",
                              youngModulus=1200, poissonRatio=0.45, totalMass=0.032)

        self.translation = translation
        self.rotation = rotation
        self.body = self.node.ElasticBody.ElasticMaterialObject

        dist = radius
        numstep = numMotors
        self.actuatedarms = []
        self.trsform = np.array([])
        for i in range(0, numstep):
            name = "ActuatedArm"+str(i)
            tr, eulerRotation = self.__getTransform(i, numstep, angleShift, radius, dist-0.8)
            arm = ActuatedArm(self.node, name=name,
                              translation=tr, eulerRotation=eulerRotation)
            self.actuatedarms.append(arm)
            # Add limits to angle that correspond to limits on real robot
            arm.ServoMotor.minAngle = -2.0225
            arm.ServoMotor.maxAngle = -0.0255
            if i == 0:
                self.trsform = [self.node.ActuatedArm0.ServoMotor.BaseFrame.dofs.position.value.copy('K')]
            else:
                angle = i*360/3
                q = Quat(self.node.ActuatedArm0.ServoMotor.BaseFrame.dofs.position.value[0][3:])
                q.rotateFromEuler([0.0, to_radians(angle), 0.0])
                pos = self.node.ActuatedArm0.ServoMotor.BaseFrame.dofs.position.value.copy('K')
                tmp = arm.ServoMotor.BaseFrame.dofs.position.value[0][:3]
                pos[0].put([0, 1, 2], tmp)
                pos[1].put([0, 1, 2], tmp)
                pos[0].put([3, 4, 5, 6], q)
                pos[1].put([3, 4, 5, 6], q)
                self.trsform = np.append(self.trsform, [pos], axis=0)

        self.node.ActuatedArm1.ServoMotor.BaseFrame.dofs.position.value = self.trsform[1]
        self.node.ActuatedArm2.ServoMotor.BaseFrame.dofs.position.value = self.trsform[2]

        self.__attachToActuatedArms(radius, numMotors, angleShift)

    def __getTransform(self, index, numstep, angleShift, radius, dist, tr=0):
        fi = float(index)
        fnumstep = float(numstep)
        angle = fi*360/fnumstep
        angle2 = fi*360/fnumstep+angleShift
        eulerRotation = [0, angle, 0]  # @todo: Assess if useful, if not remove
        translation = [dist*sin(to_radians(angle2)), -1.35+tr, dist*cos(to_radians(angle2))]  # @todo: Assess if useful, if not remove

        eulerRotation = [i+j for i, j in zip(eulerRotation, self.rotation)]  # @todo: Assess if useful, if not remove
        translation = transformPositions([translation], translation=self.translation, eulerRotation=self.rotation,
                                         scale=[1.0, 1.0, 1.0])[0]  # self.rotation

        return translation, self.rotation

    def __attachToActuatedArms(self, radius=66, numMotors=3, angleShift=180.0):
        dist = radius
        numstep = numMotors
        groupIndices = []
        frames = []

        self.body.init()

        # Rigidify the deformable part at extremity to attach arms
        for i in range(0, numstep):
            translation, eulerRotation = self.__getTransform(i, numstep, angleShift, radius, dist-2)
            box = addOrientedBoxRoi(self.node, position=self.body.dofs.rest_position.getLinkPath(),
                                    name="BoxROI"+str(i), translation=vec3.vadd(translation, [0.0, 25.0, 0.0]),
                                    eulerRotation=[0, 120*i, 0], scale=[45, 15, 30])

            box.drawBoxes.value = False
            box.init()
            groupIndices.append(list(box.indices.value))

            translation, eulerRotation = self.__getTransform(i, numstep, angleShift, radius, dist, 25)
            frames.append(translation + list(self.trsform[i][0][3:]))

        # ADD CENTER
        o = self.node.addObject("SphereROI", name="roi", template="Rigid3",
                                position=self.body.dofs.rest_position.getLinkPath(), centers=[0.0, 28, 0.0],
                                radii=[7.5], drawSphere=False)
        o.init()
        groupIndices.append(list(o.indices.value))

        translation = [0.0, 28, 0.0]
        translation = transformPositions([translation], translation=self.translation, eulerRotation=self.rotation,
                                         scale=[1.0, 1.0, 1.0])[0]
        frames.append(translation + list(self.trsform[0][0][3:]))
        for i in range(len(frames)):
            frames[i][1] += 2

        rigidifiedstruct = Rigidify(self.node, self.body, groupIndices=groupIndices, frames=frames,
                                    name="RigidifiedStructure")

        # Use this to activate some rendering on the rigidified object ######################################
        setData(rigidifiedstruct.RigidParts.dofs, showObject=False, showObjectScale=10, drawMode=2)
        # setData(rigidifiedstruct.RigidParts.RigidifiedParticules.dofs, showObject=True, showObjectScale=1,drawMode=1,
        # showColor=[1., 1., 0., 1.])
        # setData(rigidifiedstruct.DeformableParts.dofs, showObject=True, showObjectScale=1, drawEdges=1,
        # showColor=[0., 0., 1., 1.])
        #####################################################################################################

        # Attach arms
        for i in range(0, numstep):
            rigidifiedstruct.RigidParts.addObject('RestShapeSpringsForceField', name="rssff"+str(i), points=i,
                                                  external_rest_shape=self.actuatedarms[i].servoarm.dofs.getLinkPath(),
                                                  stiffness='1e16', angularStiffness='1e7')


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=[0.0, 0.0, 0.0])
    return goal_mo


def createScene(rootNode, config={"source": [0, 300, 0],
                                  "target": [0, 0, 0],
                                  "goalList": [0, 0, 0],
                                  "goal_node": 0,
                                  "goalPos": [0.0, 0.0, 0.0]}, mode='simu_and_visu'):

    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    scene = Scene(rootNode, gravity=[0.0, -9810, 0.0], dt=0.01, plugins=['SoftRobots', 'SofaPreconditioner',
                                                                         'SofaPython3', 'SofaOpenglVisual',
                                                                         'SofaMiscCollision', 'SofaBoundaryCondition',
                                                                         'SofaConstraint', 'SofaDeformable',
                                                                         'SofaEngine', 'SofaGeneralAnimationLoop',
                                                                         'SofaGeneralEngine', 'SofaGeneralRigid',
                                                                         'SofaImplicitOdeSolver', 'SofaLoader',
                                                                         'SofaMeshCollision', 'SofaMiscMapping',
                                                                         'SofaRigid', 'SofaSimpleFem'])

    # VISU ###################
    if visu:
        scene.VisualStyle.displayFlags = "showForceFields showBehavior showCollisionModels showVisualModels"

    # COLLISION ##############
    scene.addObject('FreeMotionAnimationLoop')
    scene.addObject('GenericConstraintSolver', name='solver', tolerance="1e-6", maxIterations="1000")
    scene.addObject('DefaultPipeline')
    scene.addObject('BruteForceDetection')
    scene.addObject('RuleBasedContactManager', responseParams="mu="+str(0.1), name='Response',
                    response='FrictionContact')
    scene.addObject('LocalMinDistance', alarmDistance=2, contactDistance=0.2, angleCone=0, coneFactor=0.5)
    rootNode.addObject('BackgroundSetting', color=[1, 1, 1, 1])
    # TRIPOD #################
    tripod = Tripod(scene.Modelling)

    scene.Simulation.addChild(tripod.ActuatedArm0)
    scene.Simulation.addChild(tripod.ActuatedArm1)
    scene.Simulation.addChild(tripod.ActuatedArm2)
    scene.Simulation.addChild(tripod.RigidifiedStructure)

    scene.Simulation.addObject('MechanicalMatrixMapper', name='MMM_'+tripod.name.value, template='Vec3,Rigid3',
                               object1=tripod.RigidifiedStructure.DeformableParts.getLinkPath(),
                               object2=tripod.RigidifiedStructure.RigidParts.dofs.getLinkPath(),
                               nodeToParse=tripod.ElasticBody.ElasticMaterialObject.getLinkPath())

    # scene.addObject(TripodController(name='TripodController',tripod=tripod,actuators=[tripod.ActuatedArm0,
    # tripod.ActuatedArm1, tripod.ActuatedArm2]))

    # SPHERE #################
    spheres = scene.Simulation.addChild("spheres")
    ball_mo = spheres.addObject("MechanicalObject", template="Vec3d", name="dofs", position=[0.0, 38.0, 3.5],
                                restScale="1",  reserve="0")
    spheres.addObject("UniformMass", totalMass=0.005)
    spheres.addObject("SphereCollisionModel", template="Vec3d", name="tSphereModel46",  radius=2, color=[1, 0, 0, 1])

    # MAZE ###################
    maze = scene.Simulation.addChild("MAZE")
    maze.addObject("MechanicalObject", template="Rigid3", name="dofs", position=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    maze.addObject("RigidRigidMapping", input=tripod.RigidifiedStructure.RigidParts.dofs.getLinkPath(), output='@./',
                   index=3)

    collisionmodel = CollisionMesh(maze, surfaceMeshFileName=path_mesh + "maze.stl", translation=[-51.5, 0, 51.5],
                                   rotation=[-90, 0, 0], collisionGroup=1, mappingType="RigidMapping")

    path = maze.addChild("Path")
    p_mesh = path.addObject('MeshObjLoader', filename=path_mesh+"path.obj", flipNormals=True, triangulate=True,
                            name='meshLoader', translation=[-51.5, 0, 51.5])
    p_mo = path.addObject("MechanicalObject", template="Rigid3d", name="dofs", position="@meshLoader.position",
                          showObject=True, showObjectScale=1.0)
    path.addObject("RigidRigidMapping", input=tripod.RigidifiedStructure.RigidParts.dofs.getLinkPath(), output='@./',
                   index=3)

    goal_mo = add_goal_node(rootNode)

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goal_node=config['goalList'][config['goal_node']],
                                    path_mesh=p_mesh, path_mo=p_mo, ball_mo=ball_mo))
    rootNode.addObject(goalSetter(name="GoalSetter", rootNode=rootNode, goalMO=goal_mo, goalPos=config['goalPos']))

    if visu:
        source = config["source"]
        target = config["target"]
        rootNode.addObject("LightManager")

        spotloc = [0, source[1], 0]
        rootNode.addObject("SpotLight", position=spotloc, direction=[0, -np.sign(source[1]), 0])

        rootNode.addObject("InteractiveCamera", name="camera", position=source, lookAt=target, zFar=500)

    return rootNode
