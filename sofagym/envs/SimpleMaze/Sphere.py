import Sofa
import Sofa.Core
from stlib3.scene.contactheader import ContactHeader


class Sphere(Sofa.Prefab):
    prefabParameters = [
        {'name': 'position', 'type': 'Vec3d', 'help': '', 'default': [0.0, 6.0, 2.0]},
        {'name': 'withSolver', 'type': 'bool', 'help': '', 'default': True}
    ]

    def __init__(self, *args, **kwargs):
        Sofa.Prefab.__init__(self, *args, **kwargs)

    def init(self):
        if self.withSolver.value:
            self.addObject('EulerImplicitSolver')
            self.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixd")
            self.addObject('GenericConstraintCorrection')
        self.addObject("MechanicalObject", position=self.position.value, name="sphere_mo")
        self.addObject("UniformMass", totalMass=1e-4)
        self.addObject('SphereCollisionModel', radius=2)


def createScene(rootNode):
    rootNode.gravity = [0., -9810., 0.]
    rootNode.dt = 0.01

    pluginList = ["Sofa.Component.AnimationLoop",
                  "Sofa.Component.Collision.Detection.Algorithm",
                  "Sofa.Component.Collision.Detection.Intersection",
                  "Sofa.Component.Collision.Geometry",
                  "Sofa.Component.Collision.Response.Contact",
                  "Sofa.Component.Constraint.Lagrangian.Correction",
                  "Sofa.Component.Constraint.Lagrangian.Solver",
                  "Sofa.Component.IO.Mesh", "Sofa.Component.LinearSolver.Direct",
                  "Sofa.Component.LinearSolver.Iterative", "Sofa.Component.Mass",
                  "Sofa.Component.ODESolver.Backward",
                  "Sofa.Component.SolidMechanics.Spring",
                  "Sofa.Component.StateContainer",
                  "Sofa.Component.Topology.Container.Constant",
                  "Sofa.Component.Visual"]

    rootNode.addObject('RequiredPlugin', pluginName=pluginList)

    ContactHeader(rootNode, alarmDistance=15, contactDistance=0.5, frictionCoef=0)
    rootNode.addObject('VisualStyle', displayFlags=['showCollisionModels', 'showBehavior'])
    rootNode.addObject('DefaultVisualManagerLoop')

    rootNode.addChild(Sphere(withSolver=True))

    return
