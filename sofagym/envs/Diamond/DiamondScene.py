# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ..."""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 01 2021"

import sys
import pathlib
from os.path import dirname, abspath

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()) + "/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from DiamondToolbox import rewardShaper, goalSetter

from splib3.animation import AnimationManagerController

path = dirname(abspath(__file__)) + "/mesh/"

DEFAULT_CONFIG = {
    "scene": "Diamond",
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
    "dt": 0.01,
}


from Sofa.Units.Definitions import kg, s, mm
from Sofa.Units.UnitSystem import MechanicalUnitSystem

# Units of the scene: SOFA values (gravity, stiffness, pressure, ...) are expressed in this system.
UNITS = MechanicalUnitSystem(time=s, length=mm, mass=kg)


def createScene(rootNode, config=DEFAULT_CONFIG, mode="simu_and_visu"):
    # Root node
    rootNode.addObject(
        "VisualStyle",
        displayFlags="showCollision showVisualModels showForceFields "
        "showInteractionForceFields hideCollisionModels "
        "hideBoundingCollisionModels hideWireframe",
    )

    # Required plugin
    rootNode.addObject("RequiredPlugin", name="ExternalPlugins", pluginName="SoftRobots")
    rootNode.addObject(
        "RequiredPlugin",
        name="SofaPlugins",
        pluginName=[
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
            "Sofa.Component.IntegrationScheme.Backward",
            "Sofa.Component.LinearSolver.Direct",
            "Sofa.Component.LinearSolver.Iterative",
            "Sofa.Component.Mapping.Linear",
            "Sofa.Component.Mass",
            "Sofa.Component.Setting",
            "Sofa.Component.SolidMechanics.FEM.Elastic",
            "Sofa.Component.StateContainer",
            "Sofa.Component.Topology.Container.Constant",
            "Sofa.Component.Topology.Container.Dynamic",
            "Sofa.Component.Visual",
            "Sofa.GL.Component.Shader",
        ],
    )

    rootNode.dt.value = config["dt"]
    # SOFA 26 update: 9810 mm/s^2 towards the fixed base (-z), the scene is in mm (was the default 9.81 along -y).
    rootNode.gravity.value = [0.0, 0.0, -9810.0]

    source = config["source"]
    target = config["target"]
    rootNode.addObject("LightManager")

    rootNode.addObject("SpotLight", position=source, direction=[target[i] - source[i] for i in range(len(source))])
    rootNode.addObject("InteractiveCamera", name="camera", position=source, lookAt=target, zFar=500)

    # Constraint solver, here we use a Gauss Seidel algorithm
    rootNode.addObject("FreeMotionAnimationLoop")
    rootNode.addObject("BlockGaussSeidelConstraintSolver", maxIterations=500, tolerance=1e-8)

    rootNode.addObject(AnimationManagerController(rootNode))

    # goal
    goal = rootNode.addChild("goal")
    goal.addObject("EulerImplicitIntegrationScheme", firstOrder=True)
    goal.addObject("CGLinearSolver", iterations=100, threshold=1e-5, tolerance=1e-5)
    # SOFA 26 update: the goal is drawn by its MechanicalObject; its SphereCollisionModel was only drawn (no collision
    # pipeline in the scene).
    goal_mo = goal.addObject(
        "MechanicalObject",
        name="goalMO",
        position=[0.0, 0.0, 125.0],
        showObject=True,
        drawMode=1,
        showObjectScale=5.0,
        showColor=[1.0, 1.0, 0.0, 1.0],
    )
    goal.addObject("UncoupledConstraintCorrection", defaultCompliance=1e-5)

    # Robot
    robot = rootNode.addChild("Robot")
    # The solvers
    robot.addObject("EulerImplicitIntegrationScheme")
    robot.addObject("EigenSimplicialLDLT", template="CompressedRowSparseMatrixd", name="linearsolver")

    # Load the volume mesh
    robot.addObject("MeshVTKLoader", name="loader", filename=path + "siliconeV0.vtu")
    robot.addObject("MeshTopology", src="@loader")
    robot.addObject(
        "MechanicalObject", name="tetras", template="Vec3", showIndices=False, showIndicesScale=4e-5, rx=90, dz=35
    )
    # Set the mechanical parameters
    robot.addObject("UniformMass", totalMass=0.5)
    robot.addObject("TetrahedronFEMForceField", youngModulus=180, poissonRatio=0.45)
    # Fix a part of the model
    robot.addObject("BoxROI", name="boxROI", box=[-15, -15, -40, 15, 15, 10], drawBoxes=True)
    robot.addObject("FixedProjectiveConstraint", indices="@boxROI.indices")
    robot.addObject("LinearSolverConstraintCorrection", linearSolver="@linearsolver")

    # Actuators
    actuators = robot.addChild("Actuators")
    # Points on the model where the cables are attached
    actuators.addObject(
        "MechanicalObject",
        name="actuatedPoints",
        template="Vec3",
        position=[[0, 0, 125], [0, 97, 45], [-97, 0, 45], [0, -97, 45], [97, 0, 45], [0, 0, 115]],
    )
    # Cables
    actuators.addObject(
        "CableConstraint",
        template="Vec3",
        name="north",
        indices=1,  # indice in the MechanicalObject of the corresponding attach point
        pullPoint=[0, 10, 30],  # point from where the cable is being pulled
        valueType="displacement",  # choose if you want to control the displacement or the force
        value=20,
    )  # the displacement or force to apply
    actuators.addObject("CableConstraint", template="Vec3", name="west", indices=2, pullPoint=[-10, 0, 30])
    actuators.addObject("CableConstraint", template="Vec3", name="south", indices=3, pullPoint=[0, -10, 30])
    actuators.addObject("CableConstraint", template="Vec3", name="east", indices=4, pullPoint=[10, 0, 30])
    # This component is used to map the attach points onto the FEM mesh
    actuators.addObject("BarycentricMapping", mapForces=False, mapMasses=False)

    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goalPos=config["goalPos"]))
    rootNode.addObject(goalSetter(name="GoalSetter", rootNode=rootNode, goalMO=goal_mo, goalPos=config["goalPos"]))

    return rootNode
