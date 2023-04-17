# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 01 2021"

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation


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
        tip = self.rootNode.Robot.Actuators.actuatedPoints.position[0]
        current_dist = np.linalg.norm(np.array(tip)-np.array(self.goal_pos))
        reward = max((self.prev_dist - current_dist)/self.prev_dist, 0)
        if current_dist < self.prev_dist:
            self.prev_dist = current_dist

        return min(3*reward**(1/2), 1.0), current_dist

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

        tip = self.rootNode.Robot.Actuators.actuatedPoints.position[0]
        self.init_dist = np.linalg.norm(np.array(tip)-np.array(self.goal_pos))
        self.prev_dist = self.init_dist


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

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]
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
            position[0] = self.goalPos

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
    return root.goal.goalMO.position[0]


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
    state = root.Robot.Actuators.actuatedPoints.position.value.tolist()

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

    if cost <= 5.0:
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
    cable_num, cable_disp = action_to_command(action)
    startCmd_Diamond(root, (cable_num, cable_disp), duration)


def displace(root, displacement, n_steps):
    """Change the value of the angle.

    Parameters:
    ----------
        acuator:
            The motor we consider.
        displacement: int
            The increment for the angle.

    Returns:
    -------
        None.
    """

    cable_num, cable_disp = displacement
    cables = [root.Robot.Actuators.north, root.Robot.Actuators.west, root.Robot.Actuators.south,
              root.Robot.Actuators.east]
    cable = cables[cable_num - 1]
    cable.value += cable_disp/n_steps


def startCmd_Diamond(root, displacement, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The root.
        acuator:
            The motor we consider.
        displacement: int
            The increment for the angle.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

    # Definition of the elements of the animation
    def executeAnimation(root, displacement, n_steps, factor):
        displace(root, displacement, n_steps)

    # Add animation in the scene
    root.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"root": root,
                    "displacement": displacement,
                    "n_steps": duration},
            duration=duration, mode="once"))#, realTimeClock=False


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
        return 1, 0.05
    elif action == 1:
        return 1, -0.05
    elif action == 2:
        return 2, 0.05
    elif action == 3:
        return 2, -0.05
    elif action == 4:
        return 3, 0.05
    elif action == 5:
        return 3, -0.05
    elif action == 6:
        return 4, 0.05
    elif action == 7:
        return 4, -0.05
    else:
        raise NotImplementedError("Action is not in range 0 - 7")


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
    return


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
    return
