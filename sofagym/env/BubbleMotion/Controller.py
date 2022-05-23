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


class ControllerBubbleMotion(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        if "bubblemotion" in kwargs:
            print(">>  Init bubblemotion...")
            self.bubblemotion = kwargs["bubblemotion"]
        else:
            print(">>  No bubblemotion ...")
            self.bubblemotion = None

        self.max_pressure = kwargs["max_pressure"]
        self.cavities = self.bubblemotion.cavities

        self.current_id = 0
        self.cavity = self.cavities[self.current_id]

        self.incr = 1
        print(">>  Init done.")

    def _addPressure(self, cavity, incr):
        old_value = cavity.SurfacePressureConstraint.value.value[0]
        new_value = old_value + incr
        if new_value >= 0 and new_value <= self.max_pressure:
            cavity.SurfacePressureConstraint.value.value = np.array([new_value])
        print(cavity.SurfacePressureConstraint.value.value)

    # def onAnimateBeginEvent(self, event):
    #     cavities = self.root.sceneModerator.bubblemotion.cavities
    #     max_pressure = self.root.sceneModerator.bubblemotion.max_pressure
    #
    #     cavities_pos = []
    #     for i in range(len(cavities)):
    #         cavity_value = cavities[i].SurfacePressureConstraint.value.value.tolist()[0]/max_pressure
    #         cavities_pos.append(cavity_value)
    #
    #     _sphere_pos =  self.root.sphere.MechanicalObject.position.value[0,:3].tolist()
    #     sphere_pos = [_sphere_pos[0]/10, _sphere_pos[1]/10, _sphere_pos[2]/5]
    #
    #     _goal_pos =  self.root.GoalSetter.goalPos
    #     goal_pos = [_goal_pos[0]/10, _goal_pos[1]/10, _goal_pos[2]/5]
    #
    #     state = cavities_pos+sphere_pos+goal_pos
    #     print("\n>> cavities_pos: ", cavities_pos)
    #     print(">> sphere_pos: ", sphere_pos)
    #     print(">> goal_pos: ", goal_pos)
    #     print(">> state: ", state)
    #
    #
    #
    #     goal = np.array(self.root.GoalSetter.goalPos)[:2]
    #     sphere = self.root.sphere
    #     current_sphere_pos = sphere.MechanicalObject.position.value[0,:2]
    #     dist = float(np.linalg.norm(current_sphere_pos-goal))
    #     print(">>  dist: ", dist)

    def onKeypressedEvent(self, event):
        key = event['key']

        if ord(key) == 18:  # left
            self.current_id = (self.current_id - 1) % 9
        if ord(key) == 20:  # right
            self.current_id = (self.current_id + 1) % 9

        print("Selected cavity:", self.current_id)
        self.cavity = self.cavities[self.current_id]

        if ord(key) == 19:
            self._addPressure(self.cavity, self.incr)
        if ord(key) == 21:
            self._addPressure(self.cavity, -self.incr)
