# -*- coding: utf-8 -*-
"""Controller for the Abstraction of Jimmy.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "March 8 2021"

import Sofa
import json
import numpy as np

class ControllerCartStem(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root =  kwargs["root"]
        if "cartstem" in kwargs:
            print(">>  Init cartstem...")
            self.cartstem = kwargs["cartstem"]
        else:
            print(">>  No cartstem ...")
            self.cartstem = None

        self.incr = 1000
        factor_sphere = 0.05
        self.init_sphere = -factor_sphere+ 2*factor_sphere*np.random.random()

        print(">>  Init done.")
        self.changed = 0
        self.init_reward = False

    def onAnimateBeginEvent(self, event):
        if self.changed == 2:
            pass
        elif self.changed == 0:
            with self.cartstem.stem.rateAngularDeform.rateAngularDeformMO.rest_position.writeable() as pos:
                 pos[0][1] = self.init_sphere
            self.changed = 1
        elif self.changed == 1:
            with self.cartstem.stem.rateAngularDeform.rateAngularDeformMO.rest_position.writeable() as pos:
                 pos[0][1] = 0
            self.changed = 2

        cartstem = self.root.sceneModerator.cartstem

        posCart = cartstem.cart.RigidBaseMO.position.value.tolist()[0][0]
        posTip =  self.root.sphere.MechanicalObject.position.value.tolist()[0][0]

        vCart = cartstem.cart.RigidBaseMO.velocity.value.tolist()[0][0]
        vTip =  self.root.sphere.MechanicalObject.velocity.value.tolist()[0][0]
        state = [posCart, posTip, vCart, vTip]

        print("\n>> posCart: ", [posCart])
        print(">> posTip: ", [posTip])
        print(">> vCart: ", [vCart])
        print(">> vTip: ", [vTip])
        print(state)

        if self.init_reward:
            print("\n>> Reward / dist: ", self.root.Reward.getReward())
        else:
            self.root.Reward.update()
            self.init_reward = True


    def _move(self, incr):
        cartForceField = self.cartstem.cart.ConstantForceField
        force = cartForceField.force.value.tolist()
        force[0]=incr
        cartForceField.force.value = np.array(force)


    def onKeypressedEvent(self, event):
        key = event['key']
        if ord(key) == 18:  #left
            self._move(-self.incr)
        if ord(key) == 20:  #right
            self._move(self.incr)
