# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "hunoutl"
__contact__ = "leo.hunout@inria.fr"
__version__ = "0.0.0"
__copyright__ = "(c) 2023, Inria"
__date__ = "Jan 9 2023"

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

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
        reward, cost = {0,0}

        return reward, cost

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
    state = 0
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


def action_to_command(actions, root, nb_step):
    """Link between Gym action (int) and SOFA command (displacement of cables).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).
        root:
            The root of the scene.

    Returns:
    -------
        The command.
    """

    incr = root.applyAction.compute_action(actions, nb_step)
    return incr


def startCmd(root, actions, duration):
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
    incr = action_to_command(actions, root, duration/root.dt.value + 1)
    startCmd_StemPendulum(root, incr, duration)


def startCmd_StemPendulum(rootNode, incr, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        incr:
            The elements of the commande.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

    # Definition of the elements of the animation
    def executeAnimation(rootNode, incr, factor):
        rootNode.applyAction.apply_action(incr)

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"rootNode": rootNode,
                    "incr": incr},
            duration=duration, mode="once", realTimeClock=False))




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
