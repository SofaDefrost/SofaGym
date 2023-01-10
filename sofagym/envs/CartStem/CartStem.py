# -*- coding: utf-8 -*-
"""Create the Cartstem


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")

from sofagym.utils import createCosserat as cosserat
from sofagym.utils import addRigidObject


class CartStem:
    def __init__(self, *args, **kwargs):

        if "cartstem_config" in kwargs:
            print(">>  Init cartstem_config...")
            self.cartstem_config = kwargs["cartstem_config"]

            self.init_pos = self.cartstem_config["init_pos"]
            self.cart_size = self.cartstem_config["cart_size"]
            self.max_move = self.cartstem_config["max_move"]
            self.max_v = self.cartstem_config["max_v"]  # cm/s
            self.dt = self.cartstem_config["dt"]
            self.cosserat_config = self.cartstem_config["cosserat_config"]

        else:
            print(">>  No cartstem_config ...")
            exit(1)

    def onEnd(self, rootNode):
        print(">>  Init CartStem...")
        self.cartstem = rootNode.addChild('cartstem')

        # ADD CART
        self.cart = addRigidObject(self.cartstem, filename='mesh/cube.obj', name='Cart', scale=self.cart_size,
                                   position=self.init_pos + [0, -0.7071068, 0, 0.7071068], density=1)
        self.cart.MechanicalObject.name.value = "RigidBaseMO"

        # ADD PATH
        path_config = {'init_pos': [-self.max_move, self.init_pos[1], self.init_pos[2]], 'tot_length': 2*self.max_move,
                       'nbSectionS': 1, 'nbFramesF': 1}
        self.path = cosserat(self.cartstem, path_config, name="Path", orientation=[0, 0, 0, 1], radius=0.5)

        # ADD STEM
        self.stem = cosserat(self.cartstem, self.cosserat_config, name="Stem", orientation=[0, 0, 0, 1], radius=0.5,
                             rigidBase=self.cart, buildCollision=False, youngModulus=2.7*1e6)
        id = self.cosserat_config['nbFramesF'] + 1
        self.sphere = self._createSphere(rootNode, name="sphere", scale=0.5,
                                         translation=[self.init_pos[0], self.init_pos[1], self.init_pos[2]+25],
                                         collisionGroup=0)
        rootNode.addObject('BilateralInteractionConstraint', template='Rigid3d',
                           object2=self.cart.MappedFrames.FramesMO.getLinkPath(),
                           object1=self.sphere.MechanicalObject.getLinkPath(), first_point=0, second_point=id)

        # ADD PARTIAL CONSTRAINT FOR THE CART
        self.cart.addObject('PartialFixedConstraint', fixedDirections=[0, 1, 1, 1, 1, 1])

    def _createSphere(self, parent, name="Sphere", scale=1., translation=[0, 0, 0], collisionGroup=0):
        sphere = parent.addChild(name)

        sphere.addObject('EulerImplicitSolver', firstOrder=0, rayleighStiffness=0.2, rayleighMass=0.1)
        sphere.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixd', name='solver')
        sphere.addObject('GenericConstraintCorrection')

        sphere.addObject('MechanicalObject', template="Rigid3d", translation=translation)
        sphere.addObject('UniformMass', totalMass=0.75)

        sphereVisu = sphere.addChild(name+"_VisualModel")
        sphereVisu.loader = sphereVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/ball.obj")
        sphereVisu.addObject('OglModel', name="model", src="@loader", scale3d=[scale]*3, color=[0, 1, 0, 1],
                             updateNormals=False)
        sphereVisu.addObject('RigidMapping')

        return sphere

    def getPos(self):
        posFrame = self.stem.Cart.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput = self.stem.Cart.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate = self.stem.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        posCart = self.cart.RigidBaseMO.position.value[:].tolist()
        sphere_pos = self.sphere.MechanicalObject.position.value[:].tolist()
        return [posFrame, posOutput, rate, posCart, sphere_pos]

    def setPos(self, pos):
        posFrame, posOutput, rate, posCart,sphere_pos = pos

        self.stem.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate)
        self.stem.Cart.MappedFrames.FramesMO.position.value = np.array(posFrame)
        self.stem.Cart.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput)
        self.sphere.MechanicalObject.position.value = np.array(sphere_pos)
        self.stem.Cart.MappedFrames.DiscreteCosseratMapping.init()

        self.cart.RigidBaseMO.position.value = np.array(posCart)
