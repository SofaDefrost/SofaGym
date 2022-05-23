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
import numpy as np


class ControllerCartStem(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        if "cartstem" in kwargs:
            print(">>  Init cartstem...")
            self.cartstem = kwargs["cartstem"]
        else:
            print(">>  No cartstem ...")
            self.cartstem = None

        self.incr = 0.1
        print(">>  Init done.")
        self.init_reward = False

    def onAnimateBeginEvent(self, event):
        contacts = self.root.sceneModerator.contacts
        cartstem = self.root.sceneModerator.cartstem
        factor = cartstem.max_move

        posCart = cartstem.cart.RigidBaseMO.position.value.tolist()[0][0]/factor
        posTip = cartstem.cart.MappedFrames.sphere_CollisionModel.MechanicalObject.position.value.tolist()[0][0]/factor
        posContacts = [p/factor for p in contacts.getPos()]
        goal = self.root.GoalSetter.goalPos[0]/factor
        state = [posCart, posTip] + posContacts + [goal]

        print("\n>> posCart: ", posCart)
        print(">> posTip: ", posTip)
        print(">> posContacts: ", posContacts)
        print(">> goal: ", [goal])
        print(">> MAX MOVE:", cartstem.max_move)
        print(state)

        if self.init_reward:
            print("\n>> Reward / dist: ", self.root.Reward.getReward())
            print(">> Init dist: ", self.root.Reward.init_goal_dist)
        else:
            self.root.Reward.update()
            self.init_reward = True

    def _move(self, incr):
        cartMO = self.cartstem.cart.RigidBaseMO
        pos = cartMO.position.value.tolist()
        pos[0][0]+=incr
        cartMO.position.value = np.array(pos)

    def onKeypressedEvent(self, event):
        key = event['key']
        if ord(key) == 18:  # left
            self._move(-self.incr)
        if ord(key) == 20:  # right
            self._move(self.incr)
