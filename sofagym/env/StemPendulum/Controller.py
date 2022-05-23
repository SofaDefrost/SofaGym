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

class ControllerStemPendulum(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root =  kwargs["root"]
        if "stempendulum" in kwargs:
            print(">>  Init stempendulum...")
            self.stempendulum = kwargs["stempendulum"]
        else:
            print(">>  No stempendulum ...")
            self.stempendulum = None

        self.max_torque = kwargs["max_torque"]
        self.incr = self.max_torque/10

        print(">>  Init done.")
        self.init_reward = False

    def onAnimateBeginEvent(self, event):
        stempendulum = self.root.sceneModerator.stempendulum
        beam = stempendulum.beam
        beam_len = stempendulum.beam_len

        angBase = beam.MechanicalObject.position.value.tolist()[0][-2:]
        omegaBase = beam.MechanicalObject.velocity.value.tolist()[0][-1]/30
        posTip = [p/beam_len for p in beam.MechanicalObject.position.value.tolist()[-1][:3][:-1]]

        state = angBase + [omegaBase] + posTip
        print("\n>> angBase: ", angBase)
        print(">> omegaBase: ", omegaBase)
        print(">> posTip: ", posTip)
        print(state)

        if self.init_reward:
            print("\n>> Reward : ", self.root.Reward.getReward())
        else:
            self.root.Reward.update()
            self.init_reward = True


    def _move(self, incr):
        baseForceField = self.stempendulum.beam.ConstantForceField
        force = baseForceField.force.value.tolist()

        if incr > 0:
            force[5] = self.max_torque
        else:
            force[5] = - self.max_torque
        baseForceField.force.value = np.array(force)


    def onKeypressedEvent(self, event):
        key = event['key']
        if ord(key) == 18:  #left
            self._move(-self.incr)
        if ord(key) == 20:  #right
            self._move(self.incr)
