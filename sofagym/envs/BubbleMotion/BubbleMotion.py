# -*- coding: utf-8 -*-
"""Create the StemPendulum


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

import os
import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")


class BubbleMotion:
    def __init__(self, *args, **kwargs):

        if "bubblemotion_config" in kwargs:
            print(">>  Init bubblemotion_config...")
            self.bubblemotion_config = kwargs["bubblemotion_config"]
            self.dt = self.bubblemotion_config["dt"]
            self.max_pressure = self.bubblemotion_config["max_pressure"]
            self.init_pos = self.bubblemotion_config['init_pos']

        else:
            print(">>  No bubblemotion_config ...")
            exit(1)

    def onEnd(self, rootNode):
        print(">>  Init bubblemotion...")
        self.board = self._createBoard(rootNode)

        self.cavities = []
        for i in range(9):
            cavity = self._createCavity(self.board, i, self.max_pressure)
            self.cavities.append(cavity)

        self.sphere = self._createSphere(rootNode, scale=1.0, totMass=2, translation=self.init_pos)

    def _createBoard(self, parent):
        board = parent.addChild('board')
        path = os.path.dirname(os.path.abspath(__file__))
        board.addObject('EulerImplicitSolver', name='odesolver')
        board.addObject('SparseLDLSolver', name='preconditioner', template="CompressedRowSparseMatrixd")
        board.addObject('MeshVTKLoader', name='loader', filename=path + "/mesh/Board_Volumetric.vtk")
        board.addObject('TetrahedronSetTopologyContainer', position="@loader.position", tetrahedra="@loader.tetrahedra", name='container')
        board.addObject('TetrahedronSetTopologyModifier')
        board.addObject('MechanicalObject', name='tetras', template='Vec3', rx=0, dz=0)
        board.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=0.3,
                        youngModulus=30000)

        board.addObject('BoxROI', name='boxROI_bottom', box=[0, 0, 0, 10, 10, 0.1], drawBoxes=True,
                        position="@tetras.rest_position", tetrahedra="@container.tetrahedra")
        board.addObject('RestShapeSpringsForceField', name = "RestShapeSpringsForceField_bottom",
                        points='@boxROI_bottom.indices', stiffness=1e12)

        board.addObject('BoxROI', name='boxROI_up', box=[0, 0, 1.1, 10, 10, 4], drawBoxes=True,
                        position="@tetras.rest_position", tetrahedra="@container.tetrahedra")
        board.addObject('RestShapeSpringsForceField', name="RestShapeSpringsForceField_up",
                        points='@boxROI_up.indices', stiffness=1e12)

        board.addObject('GenericConstraintCorrection')

        modelCollis = board.addChild('modelCollis')
        modelCollis.addObject('MeshSTLLoader', name='loader', filename=path + "/mesh/Board_Surface.stl")
        modelCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
        modelCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
        modelCollis.addObject('TriangleCollisionModel', group=1, color=[1, 0, 0, 0.4])
        modelCollis.addObject('BarycentricMapping',  mapForces=False, mapMasses=False)

        return board

    def _createCavity(self, parent, idx, max_pressure):
        path = os.path.dirname(os.path.abspath(__file__))
        cavity = parent.addChild('cavity_'+str(idx))
        cavity.addObject('MeshSTLLoader', name='loader', filename= path + "/mesh/Cavity_Surface_"+str(idx)+".stl")
        cavity.addObject('MeshTopology', src='@loader', name='topo')
        cavity.addObject('MechanicalObject')
        cavity.addObject('SurfacePressureConstraint', template='Vec3', triangles='@topo.triangles',
                         maxPressure=max_pressure, minPressure=0)
        cavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)
        return cavity

    def _createSphere(self, parent, name="sphere", scale=1., totMass=1, translation=[0, 0, 0]):
        path = os.path.dirname(os.path.abspath(__file__))
        filename = path+"/mesh/Ball_"

        object = parent.addChild(name)
        object.addObject('MechanicalObject', template='Rigid3', position=translation+[0, 0, 0, 1])

        object.addObject('EulerImplicitSolver')
        object.addObject('CGLinearSolver', tolerance=1e-5, iterations=25)
        object.addObject('UncoupledConstraintCorrection')

        visu = object.addChild('Visu')
        visu.addObject('MeshVTKLoader', name='loader', filename=filename+"Volumetric.vtk", scale3d=[scale]*3)
        visu.addObject('OglModel', src='@loader', texturename="", color=[0, 0, 1, 0.5])
        visu.addObject('RigidMapping')

        object.addObject('UniformMass', totalMass=totMass)

        collision = object.addChild('Collision')
        collision.addObject('MeshSTLLoader', name='loader', filename=filename+"Surface.stl", scale3d=[scale]*3)
        collision.addObject('MeshTopology', src='@loader')
        collision.addObject('MechanicalObject')
        collision.addObject('PointCollisionModel')
        collision.addObject('RigidMapping')

        return object

    def getPos(self):
        cavities_pos = []
        for i in range(9):
            cavity_pos = self.cavities[i].MechanicalObject.position.value.tolist()
            cavities_pos.append(cavity_pos)

        sphere_pos = [self.sphere.MechanicalObject.position.value.tolist()]
        board_pos = [self.board.tetras.position.value.tolist()]

        return cavities_pos + sphere_pos + board_pos

    def setPos(self, pos):
        for i in range(9):
            self.cavities[i].MechanicalObject.position.value = np.array(pos[i])
        self.sphere.MechanicalObject.position.value = np.array(pos[-2])
        self.board.tetras.position.value = np.array(pos[-1])
