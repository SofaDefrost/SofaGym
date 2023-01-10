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
from math import cos, sin

import sys
import importlib
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")

from sofagym.utils import createCosserat as cosserat
from sofagym.utils import addRigidObject

class StemPendulum():
    def __init__(self, *args, **kwargs):

        if "stempendulum_config" in kwargs:
            print(">>  Init StemPendulum_config...")
            self.stempendulum_config = kwargs["stempendulum_config"]

            self.init_or = self.stempendulum_config["init_or"]
            self.base_size = self.stempendulum_config["base_size"]
            self.max_torque = self.stempendulum_config["max_torque"]
            self.dt =  self.stempendulum_config["dt"]
            self.beam_config =  self.stempendulum_config["beam_config"]
            self.beam_len = self.beam_config['tot_length']

        else:
            print(">>  No StemPendulum_config ...")
            exit(1)


    def onEnd(self, rootNode):
        print(">>  Init StemPendulum...")
        self.stempendulum = rootNode.addChild('stempendulum')
        self._addBeam(self.stempendulum)

        #ADD PATH
        path_config = {'init_pos': [0,0,1], 'tot_length': 2, 'nbSectionS': 1, 'nbFramesF': 1}
        self.path = cosserat(self.stempendulum, path_config, name = "Path", orientation =  [ 0, 0.7071068, 0, 0.7071068 ], radius = 0.5)

        #ADD CONSTRAINT FOR THE BASE
        self.beam.addObject('PartialFixedConstraint', indices = [0], fixedDirections=[1, 1, 1, 1, 1, 0])
        self.beam.addObject('ConstantForceField', totalForce=[0, 0, 0, 0, 0, 0], indices = [0])
        self.beam.addObject('RestShapeSpringsForceField', name='spring', stiffness=0, angularStiffness=300, external_points="0", mstate=self.beam.MechanicalObject.getLinkPath(), points=0, template="Rigid3d")

    def _addBeam(self, node, collisionGroup = 0):
        x, y, z =  self.beam_config['init_pos']
        nbSections =  self.beam_config['nbSections']
        sectionSize = self.beam_config['tot_length']/nbSections

        self.beam = node.addChild("Beam")

        positions = [[sectionSize*i+x,y,z] for i in range(nbSections+1)]
        edges=[[i,i+1] for i in range(nbSections)]
        self.beam.addObject('EulerImplicitSolver')
        self.beam.addObject('SparseLDLSolver', name='solver')
        self.beam.addObject('EdgeSetTopologyContainer', position=positions, edges=edges)
        self.beam.addObject('MechanicalObject', template='Rigid3', position=[p+[0,0,0,1] for p in positions], rotation = [0, 0, self.init_or])
        self.beam.addObject('BeamInterpolation', radius=2, defaultYoungModulus=3*1e3, straight=False)
        self.beam.addObject('AdaptiveBeamForceFieldAndMass', massDensity=1e-2)
        self.beam.addObject('LinearSolverConstraintCorrection')
        self.beam.addObject('UniformMass', indices = [nbSections+1], totalMass = 1)

        topology = self.beam.addChild('Topology')
        topology.addObject('MeshObjLoader', name = "cube_topo", filename='mesh/cube.obj', scale3d=[0.2, 0.1, 0.1])
        topology.addObject('MeshObjLoader', name = "cylinder_topo", filename='mesh/cylinder.obj', scale3d=[0.5, 0.2, 0.5], rotation = [90, 0, 0])
        topology.addObject('MeshObjLoader', name = "ball_topo", filename='mesh/ball.obj', scale3d=[0.3, 0.3, 0.3])

        for i in range(nbSections+1):
            if i == nbSections:
                cube = self.beam.addChild('Ball')
                cube.addObject('TriangleSetTopologyContainer', src=topology.ball_topo.getLinkPath())
                cube.addObject('MechanicalObject')
                cube.addObject('TriangleCollisionModel', group=collisionGroup)
                cube.addObject('LineCollisionModel', group=collisionGroup)
                cube.addObject('PointCollisionModel', group=collisionGroup)
                cube.addObject('RigidMapping', index=i)
            elif i!=0:
                cube = self.beam.addChild('Cube'+str(i))
                cube.addObject('TriangleSetTopologyContainer', src=topology.cube_topo.getLinkPath())
                cube.addObject('MechanicalObject')
                cube.addObject('TriangleCollisionModel', group=collisionGroup)
                cube.addObject('LineCollisionModel', group=collisionGroup)
                cube.addObject('PointCollisionModel', group=collisionGroup)
                cube.addObject('RigidMapping', index=i)
            else:
                cylinder = self.beam.addChild('Cynlinder')
                cylinder.addObject('TriangleSetTopologyContainer', src=topology.cylinder_topo.getLinkPath())
                cylinder.addObject('MechanicalObject')
                cylinder.addObject('TriangleCollisionModel', group=collisionGroup)
                cylinder.addObject('LineCollisionModel', group=collisionGroup)
                cylinder.addObject('PointCollisionModel', group=collisionGroup)
                cylinder.addObject('RigidMapping', index=i)

        return self.beam

    def getPos(self):
        posBeam = self.beam.MechanicalObject.position.value[:].tolist()

        return [posBeam]

    def setPos(self, pos):
        [posBeam] = pos
        self.beam.MechanicalObject.position.value = np.array(posBeam)
