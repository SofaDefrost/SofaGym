# -*- coding: utf-8 -*-
"""Create the Cartstem


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")

from sofagym.utils import createCosserat as cosserat
from sofagym.utils import addRigidObject

import numpy as np

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
                                   position=self.init_pos + [0, -0.7071068, 0, 0.7071068], density=800000)
        self.cart.MechanicalObject.name.value = "RigidBaseMO"
        
        # ADD PATH
        path_config = {'init_pos': [-self.max_move, self.init_pos[1], self.init_pos[2]], 'tot_length': 2*self.max_move,
                       'nbSectionS': 1, 'nbFramesF': 1}
        self.path = cosserat(self.cartstem, path_config, name="Path", orientation=[0, 0, 0, 1], radius=0.5)

        # ADD STEM
        self.stem = cosserat(self.cartstem, self.cosserat_config, name="Stem", orientation=[0, 0, 0, 1], radius=0.5,
                             rigidBase=self.cart, buildCollision=True)
        id = self.cosserat_config['nbFramesF'] + 1
        self.sphere = self._createSphere(self.cart.MappedFrames, id, name="sphere", scale=0.5, translation=[1, 0, 0],
                                         collisionGroup=0)

        # ADD PARTIAL CONSTRAINT FOR THE CART
        self.cart.addObject('PartialFixedConstraint', fixedDirections=[0, 1, 1, 1, 1, 1])

    def _createSphere(self, parent, idx, name="Sphere", scale=1., translation=[0, 0, 0], collisionGroup=0):
        sphereVisu = parent.addChild(name+"_VisualModel")
        sphereVisu.loader = sphereVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/ball.obj")
        sphereVisu.addObject('OglModel', name="model", src="@loader", scale3d=[scale]*3, color=[0, 1, 0, 1],
                             updateNormals=False, translation=translation)
        sphereVisu.addObject('RigidMapping', index=idx)

        collision = parent.addChild(name+'_CollisionModel')
        collision.addObject('MechanicalObject', translation=translation)
        collision.addObject('PointCollisionModel', group=collisionGroup)
        collision.addObject('RigidMapping', index=idx)
        return collision

    def getPos(self):
        posFrame = self.stem.Cart.MappedFrames.FramesMO.position.value[:].tolist()
        posOutput = self.stem.Cart.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value[:].tolist()
        rate = self.stem.rateAngularDeform.rateAngularDeformMO.position.value[:].tolist()
        posCart = self.cart.RigidBaseMO.position.value[:].tolist()

        return [posFrame, posOutput, rate, posCart]

    def setPos(self, pos):
        posFrame, posOutput, rate, posCart = pos

        self.stem.rateAngularDeform.rateAngularDeformMO.position.value = np.array(rate)
        self.stem.Cart.MappedFrames.FramesMO.position.value = np.array(posFrame)
        self.stem.Cart.MappedFrames.DiscreteCosseratMapping.curv_abs_output.value = np.array(posOutput)

        self.stem.Cart.MappedFrames.DiscreteCosseratMapping.init()

        self.cart.RigidBaseMO.position.value = np.array(posCart)


class Contacts:
    def __init__(self, *args, **kwargs):

        if "contact_config" in kwargs:
            print(">>  Init contact_config...")
            self.contact_config = kwargs["contact_config"]

            self.init_pos = self.contact_config["init_pos"]
            self.cube_size = self.contact_config["cube_size"]
            self.cube_x = self.contact_config["cube_x"]
        else:
            print(">>  No contact_config ...")
            exit(1)

    def onEnd(self, rootNode):
        print(">>  Init CartStem...")
        self.contacts = rootNode.addChild('contacts')

        # ADD Cube1
        pos_Cube_1 = [p for p in self.init_pos]
        pos_Cube_1[0] = pos_Cube_1[0] + self.cube_x[0]
        self.Cube_1 = addRigidObject(self.contacts, filename='mesh/cube.obj', name='Cube_1', scale=self.cube_size,
                                     position=pos_Cube_1 + [0, 0.3826834, 0, 0.9238795])
        self.Cube_1.addObject('FixedConstraint', indices=0)

        # ADD Cube2
        pos_Cube_2 = [p for p in self.init_pos]
        pos_Cube_2[0] = pos_Cube_2[0] + self.cube_x[1]
        self.Cube_2 = addRigidObject(self.contacts, filename='mesh/cube.obj', name='Cube_2', scale=self.cube_size,
                                     position=pos_Cube_2 + [0, 0.3826834, 0, 0.9238795])
        self.Cube_2.addObject('FixedConstraint', indices=0)

    def getPos(self):
        return self.cube_x + self.cube_size
