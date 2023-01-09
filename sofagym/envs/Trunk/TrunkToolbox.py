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

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        trunkTips = self._computeTips()
        current_dist = np.linalg.norm(np.array(trunkTips)-np.array(self.goal_pos))
        reward = max((self.prev_dist - current_dist)/self.prev_dist, 0)
        self.prev_dist = current_dist

        return min(reward**(1/2), 1.0), current_dist

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

        trunkTips = self._computeTips()
        self.init_dist = np.linalg.norm(np.array(trunkTips)-np.array(self.goal_pos))
        self.prev_dist = self.init_dist

    def _computeTips(self):
        """Compute the position of the tip.

        Parameters:
        ----------
            None.

        Return:
        ------
            The position of the tip.
        """
        cables = self.rootNode.trunk.cables[:4]
        size = len(cables)

        trunkTips = np.zeros(3)
        for cable in cables:
            trunkTips += cable.meca.position[-1]/size

        return trunkTips


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


def _getGoalPos(rootNode):
    """Get XYZ position of the goal.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        The position of the goal.
    """
    return rootNode.Goal.GoalMO.position[0]


def getState(rootNode):
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

    cables = rootNode.trunk.cables[:4]
    nb_point = cables[0].meca.position.shape[0]

    points = []
    for i in range(nb_point):
        point = np.zeros(3)
        for cable in cables:
            c = cable.meca.position[i]
            point += c
        point = [round(float(k), cs)/4 for k in point]
        points += point

    goalPos = _getGoalPos(rootNode).tolist()

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

    if cost <= 1.0:
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
    num_cable, displacement = action_to_command(action)
    startCmd_Trunk(root, root.trunk.cables[num_cable], displacement, duration)


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
    elif   action == 8:
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
        raise NotImplementedError("Action must be in range 0 - 15")

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
    return root.Simulation.Trunk.dofs.position.value.tolist()


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
    root.Simulation.Trunk.dofs.position.value = np.array(pos)
