# -*- coding: utf-8 -*-
"""Create the Cartstem


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")

from sofagym.utils import createCosserat as cosserat

import numpy as np

class Cart:
    def __init__(self, *args, **kwargs):
        self.path = os.path.dirname(os.path.abspath(__file__))

        if "cart_config" in kwargs:
            print(">>  Init cart_config...")
            self.cart_config = kwargs["cart_config"]
            self.max_move = self.cart_config["max_move"]
            self.init_pos = self.cart_config["init_pos"]
            self.init_x = self.cart_config["init_x"]
        else:
            print(">>  No cart_config ...")
            exit(1)

    def onEnd(self, rootNode):
        print(">>  Init Cart ...")

        # ADD CART
        self.cart = rootNode.addChild('cart')
        init_pos = [self.init_x-3, self.init_pos[1]-2, self.init_pos[2]]
        self.cart.addObject('MechanicalObject', template='Rigid3', position=init_pos+[0, 0, 0, 1])

        self.cart.addObject('EulerImplicitSolver')
        self.cart.addObject('CGLinearSolver', tolerance=1e-5, iterations=25, threshold=1e-5)
        self.cart.addObject('UncoupledConstraintCorrection')

        self.cart.addObject('UniformMass', totalMass=1)

        self.visu = self.cart.addChild('Visu')
        self.visu.addObject('MeshVTKLoader', name='loader', filename=self.path+"/mesh/Cart_Volumetric.vtk",
                            scale3d=[1, 1, 1])
        self.visu.addObject('OglModel', src='@loader', color=[1, 0, 0, 0.5])
        self.visu.addObject('RigidMapping')

        self.collision = self.cart.addChild('Collision')
        self.collision.addObject('MeshSTLLoader', name='loader', filename=self.path+"/mesh/Cart_Surface.stl",
                                 scale3d=[1, 1, 1])
        self.collision.addObject('MeshTopology', src='@loader')
        self.collision.addObject('MechanicalObject')
        self.collision.addObject('TriangleCollisionModel', color=[1, 0, 0, 0.5])
        self.collision.addObject('RigidMapping')

        # ADD Goal
        self.goal = self.cart.addChild('Goal')
        self.goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=0.5,
                            showColor=[0, 1, 0, 0.5], translation=[3, 2, 1])
        self.goal.addObject('RigidMapping')

        # ADD PATH
        path_config = {'init_pos': [-self.max_move, self.init_pos[1], self.init_pos[2]], 'tot_length': 2*self.max_move,
                       'nbSectionS': 1, 'nbFramesF': 1}
        self.path = cosserat(rootNode, path_config, name="path", orientation=[0, 0, 0, 1], radius=0.5)

        # ADD PARTIAL CONSTRAINT FOR THE CART
        self.cart.addObject('PartialFixedConstraint', fixedDirections=[0, 1, 1, 1, 1, 1])

    def getPos(self):
        return self.cart.MechanicalObject.position.value.tolist()

    def setPos(self, pos):
        self.cart.MechanicalObject.position.value = np.array(pos)


class Ball:
    def __init__(self, *args, **kwargs):
        self.path = os.path.dirname(os.path.abspath(__file__))

        if "ball_config" in kwargs:
            print(">>  Init ball_config...")
            self.ball_config = kwargs["ball_config"]
            self.max_high = self.ball_config["max_high"]
            self.init_pos = self.ball_config["init_pos"]
            self.size_ball = self.ball_config["size_ball"]
            self.mass_ball = self.ball_config["mass_ball"]
        else:
            print(">>  No ball_config ...")
            exit(1)

    def onEnd(self, rootNode, collisionGroup=0):
        print(">>  Init Ball ...")
        self.sphere = rootNode.addChild("sphere")
        self.sphere.addObject("EulerImplicitSolver", rayleighMass=5)
        self.sphere.addObject("SparseLDLSolver", name="ldl", template="CompressedRowSparseMatrixd")
        self.sphere.addObject("GenericConstraintCorrection", solverName='@ldl')
        self.sphere.addObject("MechanicalObject", name="sphere_mo", template='Vec3', position=self.init_pos)
        self.sphere.addObject("UniformMass", totalMass=self.mass_ball)
        self.sphere.addObject("SphereCollisionModel", radius=self.size_ball, name='Sphere', color=[0, 0, 255, 255])

    def getPos(self):
        return self.sphere.sphere_mo.position.value.tolist()

    def setPos(self, pos):
        self.sphere.sphere_mo.position.value = np.array(pos)


class Gripper:
    def __init__(self, *args, **kwargs):
        self.path = os.path.dirname(os.path.abspath(__file__))

        if "gripper_config" in kwargs:
            print(">>  Init gripper_config...")
            self.gripper_config = kwargs["gripper_config"]
            self.max_pressure = self.gripper_config["max_pressure"]
        else:
            print(">>  No gripper_config ...")
            exit(1)

    def onEnd(self, rootNode, collisionGroup=0):
        print(">>  Init Gripper ...")
        self.cavities = []
        self.parts = []
        self.gripper = rootNode.addChild("gripper")

        left_part = self._createExternalPart(self.gripper, name="left_part", scale=[0.7]*3, rotation=[0, 90, 0],
                                             translation=[-1.5, -3.5, 17])
        box_pos = [-4.5, -2, 11, -4, 2, 16]
        left_part.addObject('BoxROI', name='boxROI', box=box_pos, drawBoxes=True, position="@tetras.rest_position",
                            tetrahedra="@container.tetrahedra")
        left_part.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e12)
        left_cavity = self._createCavity(left_part, self.max_pressure, scale=[0.7]*3, rotation=[0, 90, 0],
                                         translation=[-1.5, -3.5, 17])
        self.cavities.append(left_cavity)
        self.parts.append(left_part)

        right_part = self._createExternalPart(self.gripper, name="right_part", scale=[0.7]*3, rotation=[0, 90, 180],
                                              translation=[1.5, 3.5, 17])
        box_pos = [4.5, -2, 11, 4, 2, 16]
        right_part.addObject('BoxROI', name='boxROI', box=box_pos, drawBoxes=True, position="@tetras.rest_position",
                             tetrahedra="@container.tetrahedra")
        right_part.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e12)
        right_cavity = self._createCavity(right_part, self.max_pressure, scale=[0.7]*3, rotation=[0, 90, 180],
                                          translation=[1.5, 3.5, 17])
        self.cavities.append(right_cavity)
        self.parts.append(right_part)

    def _createExternalPart(self, parent, name, scale, rotation, translation):
        external_part = parent.addChild(name)
        external_part.addObject('EulerImplicitSolver', name='odesolver')
        external_part.addObject('EigenSimplicialLDLT', template='CompressedRowSparseMatrixd',name='linearsolver')
        external_part.addObject('MeshVTKLoader', name='loader', filename=self.path + "/mesh/Gripper_Volumetric.vtk",
                                scale3d=scale, rotation=rotation, translation=translation)
        external_part.addObject('TetrahedronSetTopologyContainer', position="@loader.position", tetrahedra="@loader.tetrahedra", name='container')

        external_part.addObject('TetrahedronSetTopologyModifier')
        external_part.addObject('MechanicalObject', name='tetras', template='Vec3', rx=0, dz=0)
        external_part.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large',
                                poissonRatio=0.3,  youngModulus=30000)

        external_part.addObject('SparseLDLSolver', name='preconditioner', template="CompressedRowSparseMatrixd")
        external_part.addObject('LinearSolverConstraintCorrection', solverName='@preconditioner')
        external_part.addObject('TriangleCollisionModel', group=1, color=[1, 0.2, 0.1, 0.4])
        # external_part.addObject('LineCollisionModel', group=1, color = [1, 0, 0, 0.2] )
        # external_part.addObject('PointCollisionModel', group=1, color = [1, 0, 0, 0.2] )

        return external_part

    def _createCavity(self, parent, max_pressure, scale, rotation, translation):
        cavity = parent.addChild('cavity')
        cavity.addObject('MeshSTLLoader', name='loader', filename=self.path + "/mesh/Cavity_Surface.stl", scale3d=scale,
                         rotation=rotation, translation=translation)
        cavity.addObject('MeshTopology', src='@loader', name='topo')
        cavity.addObject('MechanicalObject')
        cavity.addObject('SurfacePressureConstraint', template='Vec3', triangles='@topo.triangles',
                         maxPressure=max_pressure, minPressure=0)
        cavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)
        return cavity

    def getPos(self):

        left_part_pos = self.parts[0].tetras.position.value.tolist()
        right_part_pos = self.parts[1].tetras.position.value.tolist()

        left_cavity_pos = self.cavities[0].MechanicalObject.position.value.tolist()
        right_cavity_pos = self.cavities[1].MechanicalObject.position.value.tolist()

        return [left_part_pos, right_part_pos, left_cavity_pos, right_cavity_pos]

    def setPos(self, pos):
        [left_part_pos, right_part_pos, left_cavity_pos, right_cavity_pos] = pos
        self.parts[0].tetras.position.value = np.array(left_part_pos)
        self.parts[1].tetras.position.value = np.array(right_part_pos)

        self.cavities[0].MechanicalObject.position.value = np.array(left_cavity_pos)
        self.cavities[1].MechanicalObject.position.value = np.array(right_cavity_pos)
