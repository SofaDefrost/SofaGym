# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "emenager", "ekhairallah", "PSC"
__contact__ = "etienne.menager@ens-rennes.fr", "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime

from math import cos, sin
import numpy as np


SofaRuntime.importPlugin("SofaComponentAll")


class rewardShaper(Sofa.Core.Controller):
    """Compute the reward.

    Methods:
    -------
        __init__: Initialization of all arguments.
        getReward: Compute the reward.
        update: Initialize the value of cost.

    Arguments:
    ---------
        rootNode: <Sofa.Core>
            The scene.
        goal_pos: coordinates
            The position of the goal.
        effMO: <MechanicalObject>
            The mechanical object of the element to move.
        cost:
            Evolution of the distance between object and goal.

    """
    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]
        self.goal_pos = None
        if kwargs["goalPos"]:
            self.goal_pos = kwargs["goalPos"]
        self.effMO = None
        if kwargs["effMO"]:
            self.effMO = kwargs["effMO"]
        self.cost = None

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        current_cost = abs(self.effMO.position[0][1] - self.goal_pos[1])

        if not self.cost:
            self.cost = current_cost
            return 0, self.cost

        reward = self.cost - current_cost
        self.cost = current_cost
        return round(reward, 3)/20, self.cost

    def update(self):
        """Compute the distance between object and goal.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        self.cost = abs(self.effMO.position[0][1] - self.goal_pos[1])


class goalSetter(Sofa.Core.Controller):
    """Compute the goal.

    Methods:
    -------
        __init__: Initialization of all arguments.
        update: Initialize the value of cost.

    Arguments:
    ---------
        goalMO: <MechanicalObject>
            The mechanical object of the goal.
        goalPos: coordinates
            The coordinates of the goal.

    """

    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.goalMO = None
        if kwargs["goalMO"]:
            self.goalMO = kwargs["goalMO"]
        self.goalPos = None
        if kwargs["goalPos"]:
            self.goalPos = kwargs["goalPos"]

    def update(self):
        """Set the position of the goal.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        with self.goalMO.position.writeable() as position:
            position += self.goalPos

    def set_mo_pos(self, goal):
        """Modify the goal.

        Not used here.
        """
        pass


def _getGoalPos(root):
    """Get XYZ position of the goal.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        The position of the goal.
    """
    return root.Goal.GoalMO.position[0]


def getState(root):
    """Compute the state of the environment/agent.

    Note:
    ----
        The state is normalized.
        The state is composed of the position of the object, the displacement of
        the cable, the tips of the finger and the position of the goal.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        State: list of float
            The state of the environment/agent.
    """
    cs = 3

    cube_pos = [round(float(k), cs) for k in root.Cube.mstate.position.value.reshape(-1)]

    finger1 = root.Gripper.Finger1
    finger2 = root.Gripper.Finger2

    displacement = [float((finger1.cables.cable1.aCableActuator.getData('value').value[0]-12.5)/12.5)]

    goal_pos = _getGoalPos(root).tolist()
    goal_pos[1] /= 30

    finger_tips1 = [round(k, cs) for k in finger1.tetras.position[648].tolist()]
    finger_tips1[0] /= 120
    finger_tips1[1] /= 140
    finger_tips1[2] /= 120

    finger_tips2 = [round(k, cs) for k in finger2.tetras.position[648].tolist()]
    finger_tips2[0] /= 120
    finger_tips2[1] /= 140
    finger_tips2[2] /= 120

    finger_tips = [(finger_tips1[i] + finger_tips2[i])/2 for i in range(3)]

    state = cube_pos + displacement + finger_tips + goal_pos

    return state


def getReward(root):
    """Compute the reward using Reward.getReward().

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        done, reward

    """

    reward, cost = root.Reward.getReward()

    if reward >= 1.0:
        reward = 1.0
    elif reward < 0.0:
        reward = 0.0

    if cost <= 3:
        reward += 1
        return True, reward

    return False, reward


def getTranslated(points, vec):
    """Translate a point.

    Parameters:
    ----------
        points: list
            List of points [x, y, z]
        vec: [vec_x, vec_y, vec_z]
            Translation vector.
    """
    r = []

    for v in points:
        x = v[0]+vec[0]
        y = v[1]+vec[1]
        z = v[2]+vec[2]

        r.append([x, y, z])

        if not (abs(x)<120 and abs(y) < 140 and abs(z) < 120):
            return None

    return r


def rotate_y(point, angle, rotation_center):
    """Rotate a point relatively to a center of rotation wrt y.

    Parameters:
    ----------
        point:coordinates
            The coordinates of the original point.
        angle: float
            The rotation angle.
        rotationCenter: coordinates
            The coordinates of the rotation center.

        Returns:
        -------
            The new coordinates of the point.

    """
    translated = [point[0]-rotation_center[0], point[1]-rotation_center[1], point[2]-rotation_center[2]]

    rotated = [translated[0]*cos(angle)+translated[2]*sin(angle),
               translated[1],
               -translated[0]*sin(angle)+translated[2]*cos(angle)]

    return [rotated[0]+rotation_center[0], rotated[1]+rotation_center[1], rotated[2]+rotation_center[2]]


def getRotated(points, angle, rotation_center):
    """Rotate points relatively to a center of rotation wrt y.

    Parameters:
    ----------
        point:coordinates
            The coordinates of the original points.
        angle: float
            The rotation angle.
        rotationCenter: coordinates
            The coordinates of the rotation center.

        Returns:
        -------
            The new coordinates of the points.

    """
    r = []

    for v in points:
        r.append(rotate_y(v, angle, rotation_center))

    return r


def translateFingers(fingers, direction):
    """Function to translate finger.

    Parameters:
    ----------
        fingers: list
            The fingers.
        direction: [vec_x, vec_y, vec_z]
            Translation vector.

    Returns:
    -------
        None.

    """
    res_list = []
    possible = True

    for finger in fingers:
        mecaobject = finger.tetras
        res = getTranslated(mecaobject.position.value,  direction)
        res_list.append(res)
        if res is None:
            possible = False

    if possible:

        for i, finger in enumerate(fingers):
            mecaobject = finger.tetras
            mecaobject.position.value = res_list[i]

            cable = finger.cables.cable1.aCableActuator
            p = cable.pullPoint
            cable.pullPoint.value = [p[0]+direction[0], p[1]+direction[1], p[2]+direction[2]]


def getRotationCenter(fingers):
    """Find the rotation center.

    Parameters:
    ----------
        fingers: list
            The fingers.

    Returns:
    -------
        The rotation center.

    """
    rotation_center = [0, 0, 0]

    for finger in fingers:
        cable = finger.cables.cable1.aCableActuator
        p = cable.pullPoint
        rotation_center = [rotation_center[0] + p[0]/len(fingers),
                           rotation_center[1] + p[1]/len(fingers),
                           rotation_center[2] + p[2]/len(fingers)]
    return rotation_center


def rotateFingers(fingers, rot):
    """Function to rotate finger.

    Parameters:
    ----------
        fingers: list
            The fingers.
        rot: float
            The rotation angle.

    Returns:
    -------
        None.

    """
    rotation_center = getRotationCenter(fingers)

    for finger in fingers:
        mecaobject = finger.tetras
        mecaobject.getData('rest_position').value = getRotated(mecaobject.getData('rest_position').value, rot,
                                                               rotation_center)
        cable = finger.cables.cable1.aCableActuator
        p = cable.pullPoint
        cable.getData("pullPoint").value = rotate_y(p, rot, rotation_center)


def displace(finger, displacement):
    """Change the value of the cable in the finger.

    Parameters:
    ----------
        fingers:
            The finger.
        displacement: float
            The displacement.

    Returns:
    -------
        None.

    """
    cable = finger.cables.cable1.aCableActuator.getData('value')

    displacement = cable.value[0] + displacement

    if displacement < 0:
        displacement = 0
    if displacement > 25:
        displacement = 25
    cable.value = [displacement]


def getPos(root):
    """Return the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.

    Returns:
    -------
        The position(s) of the object(s) of the scene.
    """
    finger1_pos = root.Gripper.Finger1.tetras.position.value.tolist()
    finger2_pos = root.Gripper.Finger2.tetras.position.value.tolist()
    cube_pose = root.Cube.mstate.position.value.tolist()
    return [finger1_pos, finger2_pos, cube_pose]


def setPos(root, pos):
    """Set the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.
        pos: list
            The position(s) of the object(s) of the scene.

    Returns:
    -------
        None.

    Note:
    ----
        Don't forget to init the new value of the position.

    """
    [finger1_pos, finger2_pos, cube_pose] = pos
    root.Gripper.Finger1.tetras.position.value = np.array(finger1_pos)
    root.Gripper.Finger2.tetras.position.value = np.array(finger2_pos)
    root.Cube.mstate.position.value = np.array(cube_pose)
