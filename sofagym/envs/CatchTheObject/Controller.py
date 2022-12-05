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


class ControllerCatchTheObject(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        if "cart" in kwargs:
            print(">>  Init cart...")
            self.cart = kwargs["cart"]
        else:
            print(">>  No cart ...")
            self.cart = None

        if "gripper" in kwargs:
            print(">>  Init gripper...")
            self.gripper = kwargs["gripper"]
        else:
            print(">>  No gripper ...")
            self.gripper = None

        if "ball" in kwargs:
            print(">>  Init ball...")
            self.ball = kwargs["ball"]
        else:
            print(">>  No ball ...")
            self.ball = None

        self.max_move = self.cart.max_move
        self.max_pressure = self.gripper.max_pressure
        self.incr = 0.3
        self.direction = np.random.randint(0, 1)
        print(">>  Init done.")

    def _moveCart(self, incr):
        cartMO = self.cart.cart.MechanicalObject
        pos = cartMO.position.value.tolist()
        if abs(pos[0][0]+3+incr) <= self.max_move:
            pos[0][0] += incr
        else:
            self.direction = (self.direction+1) % 2
        cartMO.position.value = np.array(pos)

    def onAnimateBeginEvent(self, event):
        if self.direction == 1:
            self._moveCart(self.incr)
        else:
            self._moveCart(-self.incr)

        pos_ball = self.ball.sphere.sphere_mo.position.value.tolist()[0][2]/self.ball.max_high
        v_ball = self.ball.sphere.sphere_mo.velocity.value.tolist()[0][2]/175
        pos_cart = self.cart.cart.MechanicalObject.position.value.tolist()[0][0]/self.cart.max_move
        d_cart = (self.direction - 0.5)*2
        pressure = self.gripper.cavities[0].SurfacePressureConstraint.value.value.tolist()[0]/self.gripper.max_pressure

        state = [pos_ball, v_ball, pos_cart, pressure, d_cart]

        print("\n>> pos_ball: ", pos_ball)
        print(">> v_ball: ", v_ball)
        print(">> pos_cart: ", pos_cart)
        print(">> d_cart: ", d_cart)
        print(">> pressure: ", pressure)
        print(">> state: ", state)

        pos_ball = self.ball.sphere.sphere_mo.position.value[0]
        pos_goal = self.cart.cart.Goal.GoalMO.position.value[0][:3]
        max_dist = np.linalg.norm(np.array([0, self.ball.max_high])-np.array([self.cart.max_move, 0]))
        dist = np.linalg.norm(pos_ball-pos_goal)
        r = -float(dist)/max_dist
        print(">> reward: ", r, " - dist: ", dist, " - under: ", pos_ball[2] < 0)

    def _addPressure(self, incr):
        cavities = self.gripper.cavities
        for cavity in cavities:
            old_value = cavity.SurfacePressureConstraint.value.value[0]
            new_value = old_value + incr
            if new_value >= 0 and new_value <= self.max_pressure:
                cavity.SurfacePressureConstraint.value.value = np.array([new_value])

    def onKeypressedEvent(self, event):
        key = event['key']
        if ord(key) == 18:  # left
            self._addPressure(-self.incr)
        if ord(key) == 20:  # right
            self._addPressure(self.incr)
