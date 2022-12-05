# -*- coding: utf-8 -*-

import Sofa.Core
import Sofa.Simulation
from math import cos, sin, pi

from GripperToolbox import translateFingers, getRotationCenter


def rotate_x(point, angle, rotationCenter):
    translated = [point[0]-rotationCenter[0], point[1]-rotationCenter[1], point[2]-rotationCenter[2]]
    rotated = [translated[0],
               translated[1]*cos(angle)-translated[2]*sin(angle),
               translated[1]*sin(angle)+translated[2]*cos(angle)]
    return [rotated[0]+rotationCenter[0], rotated[1]+rotationCenter[1], rotated[2]+rotationCenter[2]]


def rotate_y(point, angle, rotationCenter):
    translated = [point[0]-rotationCenter[0], point[1]-rotationCenter[1], point[2]-rotationCenter[2]]
    rotated = [translated[0]*cos(angle)+translated[2]*sin(angle),
               translated[1],
               -translated[0]*sin(angle)+translated[2]*cos(angle)]
    return [rotated[0]+rotationCenter[0], rotated[1]+rotationCenter[1], rotated[2]+rotationCenter[2]]


def rotate_z(point, angle, rotationCenter):
    translated = [point[0]-rotationCenter[0], point[1]-rotationCenter[1], point[2]-rotationCenter[2]]
    rotated = [translated[0]*cos(angle)-translated[1]*sin(angle),
               translated[0]*sin(angle)+translated[1]*cos(angle),
               translated[2]]
    return [rotated[0]+rotationCenter[0], rotated[1]+rotationCenter[1], rotated[2]+rotationCenter[2]]


def rotateFingers(fingers, rotate, rot):
    rotationCenter = getRotationCenter(fingers)
    for finger in fingers:
        mecaobject = finger.tetras
        mecaobject.getData('rest_position').value = getRotated(rotate, mecaobject.getData('rest_position').value, rot,
                                                               rotationCenter)

        cable = finger.cables.cable1.aCableActuator
        p = cable.pullPoint
        cable.getData("pullPoint").value = rotate(p, rot, rotationCenter)


def getRotated(rotate, points, angle, rotationCenter):
    r = []
    for v in points:
        r.append(rotate(v, angle, rotationCenter))
    return r


class GripperController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self)
        self.fingers = kwargs['fingers']
        self.name = "GripperController"
        self.rootNode = kwargs['rootNode']
        self.N = 0
        self.i = 0
        self.errorPlot = []
        self.flag = True

    def onBeginAnimationStep(self, deltaTime):
        pass

    def onKeypressedEvent(self, k):
        c = k['key']

        rot = None
        rotate = None
        direction = None

        if c == 'C':
            rot = 1/(2*pi)
            rotate = rotate_y
        elif c == 'A':
            rot = -1/(2*pi)
            rotate = rotate_y
        elif c == '5':
            rot = 1/(2*pi)
            rotate = rotate_x
        elif c == '6':
            rot = -1/(2*pi)
            rotate = rotate_x
        elif c == '7':
            rot = 1/(2*pi)
            rotate = rotate_z
        elif c == '8':
            rot = -1/(2*pi)
            rotate = rotate_z
        elif c == 'U':
            direction = [0.0, 1.0, 0.0]
        elif c == 'D':
            direction = [0, -1, 0]
        elif ord(c) == 18:
            direction = [1.0, 0.0, 0.0]
        elif ord(c) == 20:
            direction = [-1.0, 0.0, 0.0]
        elif ord(c) == 19:
            direction = [0.0, 0.0, 1.0]
        elif ord(c) == 21:
            direction = [0.0, 0.0, -1.0]

        if rot is not None:
            rotateFingers(self.fingers, rotate, rot)

        if direction is not None:
            translateFingers(self.fingers, direction)


class FingerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self)
        self.name = 'Controller_' + kwargs['name']
        self.fingerName = kwargs['name']
        self.control1 = kwargs['control1']
        self.control2 = kwargs['control2']
        self.node = kwargs['node']
        self.cable = self.node.cables.cable1.aCableActuator.getData('value')

    def onKeypressedEvent(self, k):
        c = k['key']

        if c == self.control1:
            self.cable.value = [self.cable.value[0] + 1.]

        elif c == self.control2:
            displacement = self.cable.value[0] - 1.
            if displacement < 0:
                displacement = 0.
            self.cable.value = [displacement]
