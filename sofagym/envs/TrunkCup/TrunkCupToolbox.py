# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib.animation.animate import Animation

SofaRuntime.importPlugin("Sofa.Component")


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

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        bary = self._computeCupBary()
        current_dist = np.sum(np.abs(bary-self.goal_pos))

        return self.init_dist - current_dist, current_dist

    def update(self):
        """Update function.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """

        bary = self._computeCupBary()
        self.init_dist = np.sqrt(np.sum((bary-self.goal_pos)**2))

    def _computeCupBary(self):
        """Compute the position of the tip.

        Parameters:
        ----------
            None.

        Return:
        ------
            The position of the tip.
        """
        cup = self.rootNode.cylinder.cylinderEffector.effectorPoint.position.value
        size = cup.shape[0]
        bary = np.zeros(3)
        for point in cup:
            bary += point/size

        return bary


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

    cable_path = root.trunk
    cables = [cable_path.cableL0, cable_path.cableL1, cable_path.cableL2, cable_path.cableL3]
    nb_point = cables[0].meca.position.shape[0]

    points= []
    for i in range(nb_point):
        point = np.zeros(3)
        for cable in cables:
            c = cable.meca.position[i]
            point += c
        point = [round(float(k), cs)/4 for k in point]
        points += point

    goalPos = _getGoalPos(root).tolist()

    state = points + goalPos

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

    if cost <= 3:
        reward += 1
        return True, reward

    return False, reward


def startCmd(root, action, duration):
    """Initialize the command from root and action.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        action: int
            The action.
        duration: float
            Duration of the animation.

    Returns:
    ------
        None.

    """
    cable_path = root.trunk
    cables = [cable_path.cableL0, cable_path.cableL1, cable_path.cableL2, cable_path.cableL3,
              cable_path.cableS0, cable_path.cableS1, cable_path.cableS2, cable_path.cableS3]

    num_cable, displacement = action_to_command(action)
    startCmd_Trunk(root, cables[num_cable], displacement, duration)


def displace(cable, displacement):
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
    cable.cable.value = [cable.cable.value[0] + displacement]


def startCmd_Trunk(rootNode, cable, displacement, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        cable: <MechanicalObject>
            The mechanical object of the cable to move.
        displacement: float
            The elements of the commande.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

    # Definition of the elements of the animation
    def executeAnimation(cable, displacement, factor):
        displace(cable, displacement)

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"cable": cable,
                    "displacement": displacement},
            duration=duration, mode="once", realTimeClock=False))


def action_to_command(action):
    """Link between Gym action (int) and SOFA command (displacement of cables).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).

    Returns:
    -------
        The command (number of the cabl and its displacement).
    """
    if action == 0:
        num_cable, displacement = 0, 1
    elif action == 1:
        num_cable, displacement = 1, 1
    elif action == 2:
        num_cable, displacement = 2, 1
    elif action == 3:
        num_cable, displacement = 3, 1
    elif action == 4:
        num_cable, displacement = 4, 1
    elif action == 5:
        num_cable, displacement = 5, 1
    elif action == 6:
        num_cable, displacement = 6, 1
    elif action == 7:
        num_cable, displacement = 7, 1
    elif action == 8:
        num_cable, displacement = 0, -1
    elif action == 9:
        num_cable, displacement = 1, -1
    elif action == 10:
        num_cable, displacement = 2, -1
    elif action == 11:
        num_cable, displacement = 3, -1
    elif action == 12:
        num_cable, displacement = 4, -1
    elif action == 13:
        num_cable, displacement = 5, -1
    elif action == 14:
        num_cable, displacement = 6, -1
    elif action == 15:
        num_cable, displacement = 7, -1
    else:
        raise NotImplementedError("Action should be in range 0 - 15")

    return num_cable, displacement


def getPos(root):
    """Retun the position of the mechanical object of interest.

    Parameters:
    ----------
        root: <Sofa root>
            The root of the scene.

    Returns:
    -------
        _: list
            The position(s) of the object(s) of the scene.
    """
    trunk = root.trunk.tetras.position.value.tolist()
    cup = root.cylinder.tetras.position.value.tolist()
    return [trunk, cup]


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
    [trunk, cup] = pos
    root.trunk.tetras.position.value = np.array(trunk)
    root.cylinder.tetras.position.value = np.array(cup)
