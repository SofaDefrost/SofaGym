
from math import pi, cos, sin, pi
import random


import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from Finger import Finger


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



def rotatePullPointXYZ(pullpoint, rot_x, rot_y, rot_z):
    pullpointX = rotate_x(pullpoint, rot_x, [0, 0, 0])
    pullpointXY = rotate_y(pullpointX, rot_y, [0, 0, 0])
    pullpointXYZ = rotate_z(pullpointXY, rot_z, [0, 0, 0])
    return pullpointXYZ


def Gripper(rootNode, visu, simu):
    selfNode = rootNode.addChild("Gripper")

    translate_gripper = 0
    x_trans = 20*random.random()*random.choice([-1, 1])
    z_trans = 30*random.random()*random.choice([-1, 1])

    pullpoint = [-40, -11, 0]
    fixingBox=[-45 + x_trans, 20 + translate_gripper, -45 + z_trans, 45 + x_trans, 40 + translate_gripper, 45 + z_trans]

    translation1 = [45 + x_trans, 0 + translate_gripper, z_trans]
    rot1 = -65
    rot1rad = rot1*pi/180
    pullpoint1 = rotatePullPointXYZ(pullpoint, 0, 0, rot1rad)
    pullpoint1 = [pullpoint1[0]+translation1[0], pullpoint1[1]+translation1[1], pullpoint1[2]+translation1[2]]
    f1 = Finger(selfNode, fixingBox=fixingBox, visu=visu, simu=simu, pullPointLocation=pullpoint1, control1='1',
                control2='2', name="Finger1", rotation=[0, 0, rot1], translation=translation1)

    translation2 = [-45 + x_trans, translate_gripper, z_trans]

    rot2_x = 180
    rot2rad_x = rot2_x*pi/180

    rot2_z = 180+65
    rot2rad_z = rot2_z*pi/180
    pullpoint2 = rotatePullPointXYZ(pullpoint, rot2rad_x, 0, rot2rad_z)
    pullpoint2 = [pullpoint2[0]+translation2[0], pullpoint2[1]+translation2[1], pullpoint2[2]+translation2[2]]

    Finger(selfNode, fixingBox=fixingBox, visu=visu, simu=simu, pullPointLocation=pullpoint2, control1='1',
           control2='2', name="Finger2", rotation=[rot2_x, 0, rot2_z], translation=translation2)

    # selfNode.addObject(GripperController(node=selfNode, fingers=[f1, f2], rootNode=rootNode))

    return selfNode
