# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ..."""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 01 2021"


from os.path import dirname, abspath

path = dirname(abspath(__file__)) + "/mesh/"


from Sofa.Units.Definitions import kg, s, mm
from Sofa.Units.UnitSystem import MechanicalUnitSystem

# Units of the scene: SOFA values (gravity, stiffness, pressure, ...) are expressed in this system.
UNITS = MechanicalUnitSystem(time=s, length=mm, mass=kg)


def createScene(rootNode):
    # SOFA 26 update: 9810 mm/s^2 towards the fixed base (-z), the scene is in mm (was the default 9.81 along -y).
    rootNode.gravity.value = [0.0, 0.0, -9810.0]
    # Root node
    rootNode.addObject(
        "VisualStyle",
        displayFlags="showCollision showVisualModels showForceFields "
        "showInteractionForceFields hideCollisionModels "
        "hideBoundingCollisionModels hideWireframe",
    )

    # Required plugin
    rootNode.addObject(
        "RequiredPlugin",
        pluginName=[
            "SoftRobots",
            "SoftRobots.Inverse",
            "Sofa.Component.AnimationLoop",
            "Sofa.Component.Collision.Detection.Algorithm",
            "Sofa.Component.Collision.Detection.Intersection",
            "Sofa.Component.Collision.Geometry",
            "Sofa.Component.Collision.Response.Contact",
            "Sofa.Component.Constraint.Lagrangian.Correction",
            "Sofa.Component.Constraint.Projective",
            "Sofa.Component.Engine.Select",
            "Sofa.Component.IntegrationScheme.Backward",
            "Sofa.Component.IO.Mesh",
            "Sofa.Component.LinearSolver.Direct",
            "Sofa.Component.LinearSolver.Iterative",
            "Sofa.Component.LinearSystem",
            "Sofa.Component.Mapping.Linear",
            "Sofa.Component.Mass",
            "Sofa.Component.Setting",
            "Sofa.Component.SolidMechanics.FEM.Elastic",
            "Sofa.Component.StateContainer",
            "Sofa.Component.Topology.Container.Dynamic",
            "Sofa.Component.Visual",
        ],
    )

    rootNode.addObject("FreeMotionAnimationLoop")
    rootNode.addObject("CollisionPipeline", verbose=False)
    rootNode.addObject("BruteForceBroadPhase")
    rootNode.addObject("BVHNarrowPhase")
    rootNode.addObject("CollisionResponse", response="FrictionContactConstraint")
    rootNode.addObject("LocalMinDistance", name="Proximity", alarmDistance=3.0, contactDistance=0.5)
    rootNode.addObject("QPInverseProblemSolver", name="QP", printLog=False)
    rootNode.addObject("BackgroundSetting", color=[0.0, 0.168627, 0.211765, 1.0])

    # goal
    goal = rootNode.addChild("goal")
    goal.addObject("EulerImplicitIntegrationScheme", firstOrder=True)
    goal.addObject("CGLinearSolver", iterations=100, threshold=1e-5, tolerance=1e-5)
    goal.addObject("MechanicalObject", name="goalMO", position=[30.0, 0.0, 125.0])
    goal.addObject("SphereCollisionModel", radius=5.0, group="1")
    goal.addObject("UncoupledConstraintCorrection", defaultCompliance=1e-5)

    # feuille
    feuille = rootNode.addChild("feuille")
    feuille.addObject("EulerImplicitIntegrationScheme")
    # SOFA 26 update: PCGLinearSolver.update_step was removed, its role is taken by assemblingRate on a
    # PreconditionedMatrixFreeSystem.
    feuille.addObject("PreconditionedMatrixFreeSystem", assemblingRate=1)
    # SOFA 26 update: the preconditioner must exist before PCGLinearSolver links to it.
    # The assembled system of the preconditioner, otherwise added by SOFA (the node has a matrix-free system).
    feuille.addObject("MatrixLinearSystem", name="preconditionerSystem", template="CompressedRowSparseMatrixd")
    feuille.addObject(
        "SparseLDLSolver",
        name="preconditioner",
        template="CompressedRowSparseMatrixd",
        linearSystem="@preconditionerSystem",
    )
    feuille.addObject(
        "PCGLinearSolver",
        iterations=1,
        name="linearsolver",
        tolerance=1e-5,
        preconditioner="@preconditioner",
        use_precond=True,
    )
    feuille.addObject("MeshVTKLoader", name="loader", filename=path + "siliconeV0.vtu")
    feuille.addObject("TetrahedronSetTopologyContainer", src="@loader")
    feuille.addObject("TetrahedronSetGeometryAlgorithms", drawTetrahedra=False, template="Vec3")
    feuille.addObject(
        "MechanicalObject", name="tetras", template="Vec3", showIndices=False, showIndicesScale=4e-5, rx=90, dz=35
    )
    feuille.addObject("UniformMass", totalMass=0.5)
    feuille.addObject("TetrahedronFEMForceField", youngModulus=180, poissonRatio=0.45)
    feuille.addObject("BoxROI", name="boxROI", box=[[-15, -15, -40], [15, 15, 10]], drawBoxes=True)
    feuille.addObject("FixedProjectiveConstraint", indices="@boxROI.indices")
    feuille.addObject("LinearSolverConstraintCorrection", linearSolver="@preconditioner")

    # feuille/controlledPoints
    controlledPoints = feuille.addChild("controlledPoints")
    controlledPoints.addObject(
        "MechanicalObject",
        name="actuatedPoints",
        template="Vec3",
        position=[[0, 0, 125], [0, 97, 45], [-97, 0, 45], [0, -97, 45], [97, 0, 45], [0, 0, 115]],
    )

    controlledPoints.addObject(
        "PositionEffector", template="Vec3", indices=0, effectorGoal="@../../goal/goalMO.position"
    )

    controlledPoints.addObject(
        "CableActuator", template="Vec3", name="nord", indices=1, pullPoint=[0, 10, 30], maxPositiveDisp=20, minForce=0
    )
    controlledPoints.addObject(
        "CableActuator",
        template="Vec3",
        name="ouest",
        indices=2,
        pullPoint=[-10, 0, 30],
        maxPositiveDisp=20,
        minForce=0,
    )
    controlledPoints.addObject(
        "CableActuator", template="Vec3", name="sud", indices=3, pullPoint=[0, -10, 30], maxPositiveDisp=20, minForce=0
    )
    controlledPoints.addObject(
        "CableActuator", template="Vec3", name="est", indices=4, pullPoint=[10, 0, 30], maxPositiveDisp=20, minForce=0
    )

    controlledPoints.addObject("BarycentricMapping", mapForces=False, mapMasses=False)

    return rootNode
