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
from os.path import dirname, abspath

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from DiamondToolbox import rewardShaper, goalSetter

from splib3.animation import AnimationManagerController

path = dirname(abspath(__file__))+'/mesh/'

DEFAULT_CONFIG = {"scene": "Diamond",
                  "deterministic": True,
                  "source": [0, 200, 0],
                  "target": [0, 0, 0],
                  "goalList": [[30.0, 0.0, 125.0], [-30.0, 0.0, 125.0], [0.0, 30.0, 125.0], [0.0, -30.0, 125.0]],
                  "goalPos": [30.0, 0.0, 125.0],
                  "scale_factor": 200,
                  "timer_limit": 50,
                  "timeout": 30,
                  "display_size": (1600, 800),
                  "render": 1,
                  "save_data": False,
                  "save_path": path + "/Results" + "/Diamond",
                  "planning": True,
                  "discrete": True,
                  "seed": 0,
                  "start_from_history": None,
                  "python_version": "python3.7",
                  "zFar": 5000,
                  }


def createScene(rootNode, config=DEFAULT_CONFIG, mode='simu_and_visu'):
    # Root node
    rootNode.addObject('VisualStyle', displayFlags='showCollision showVisualModels showForceFields '
                                                   'showInteractionForceFields hideCollisionModels '
                                                   'hideBoundingCollisionModels hideWireframe')

    # Required plugin
    rootNode.addObject('RequiredPlugin', name='ExternalPlugins', pluginName='SoftRobots')
    rootNode.addObject('RequiredPlugin', name='SofaPlugins', pluginName=['SofaBoundaryCondition',
                                                                         'SofaImplicitOdeSolver',
                                                                         'SofaPreconditioner',
                                                                         'SofaSimpleFem',
                                                                         'SofaSparseSolver',
                                                                         'SofaLoader',
                                                                         'SofaEngine',
                                                                         'SofaConstraint'])

    source = config["source"]
    target = config["target"]
    rootNode.addObject("LightManager")

    rootNode.addObject("SpotLight", position=source, direction=[target[i] - source[i] for i in range(len(source))])
    rootNode.addObject("InteractiveCamera", name='camera', position=source, lookAt=target, zFar=500)

    # Constraint solver, here we use a Gauss Seidel algorithm
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', maxIterations=500, tolerance=1e-8)

    rootNode.addObject(AnimationManagerController(rootNode))

    # goal
    goal = rootNode.addChild('goal')
    goal.addObject('EulerImplicitSolver', firstOrder=True)
    goal.addObject('CGLinearSolver', iterations=100, threshold=1e-5, tolerance=1e-5)
    goal_mo = goal.addObject('MechanicalObject', name='goalMO', position=[0.0, 0.0, 125.0])
    goal.addObject('SphereCollisionModel', radius=5.0, group='1')
    goal.addObject('UncoupledConstraintCorrection')

    # Robot
    robot = rootNode.addChild('Robot')
    # The solvers
    robot.addObject('EulerImplicitSolver')
    robot.addObject('EigenSimplicialLDLT',template='CompressedRowSparseMatrixd', name="linearsolver")
    
    # Load the volume mesh
    robot.addObject('MeshVTKLoader', name="loader", filename=path+'siliconeV0.vtu')
    robot.addObject('MeshTopology', src="@loader")
    robot.addObject('MechanicalObject', name="tetras", template="Vec3", showIndices=False, showIndicesScale=4e-5,
                    rx=90, dz=35)
    # Set the mechanical parameters
    robot.addObject('UniformMass', totalMass=0.5)
    robot.addObject('TetrahedronFEMForceField', youngModulus=180, poissonRatio=0.45)
    # Fix a part of the model
    robot.addObject('BoxROI', name="boxROI", box=[-15, -15, -40,  15, 15, 10], drawBoxes=True)
    robot.addObject('FixedConstraint', indices="@boxROI.indices")
    robot.addObject('LinearSolverConstraintCorrection', solverName="@preconditioner")

    # Actuators
    actuators = robot.addChild('Actuators')
    # Points on the model where the cables are attached
    actuators.addObject('MechanicalObject', name="actuatedPoints", template="Vec3",
                        position=[[0, 0, 125], [0, 97, 45], [-97, 0, 45], [0, -97, 45], [97, 0, 45], [0, 0, 115]])
    # Cables
    actuators.addObject('CableConstraint', template="Vec3", name="north",
                        indices=1,  # indice in the MechanicalObject of the corresponding attach point
                        pullPoint=[0, 10, 30],  # point from where the cable is being pulled
                        valueType='displacement',  # choose if you want to control the displacement or the force
                        value=20)  # the displacement or force to apply
    actuators.addObject('CableConstraint', template="Vec3", name="west",  indices=2, pullPoint=[-10, 0, 30])
    actuators.addObject('CableConstraint', template="Vec3", name="south", indices=3, pullPoint=[0, -10, 30])
    actuators.addObject('CableConstraint', template="Vec3", name="east",  indices=4, pullPoint=[10, 0, 30])
    # This component is used to map the attach points onto the FEM mesh
    actuators.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goalPos=config['goalPos']))
    rootNode.addObject(goalSetter(name="GoalSetter", rootNode=rootNode, goalMO=goal_mo, goalPos=config['goalPos']))

    return rootNode
