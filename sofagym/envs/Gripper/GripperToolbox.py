# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "emenager", "ekhairallah"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import SofaRuntime
from splib3.animation.animate import Animation

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from GripperTools import rewardShaper, goalSetter, _getGoalPos, getState, getReward, \
    getRotationCenter, translateFingers, rotateFingers, displace, getPos, setPos


SofaRuntime.importPlugin("Sofa.Component")


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
    rotation, direction, displacement = action_to_command(action)
    startCmd_Gripper(root, [root.Gripper.Finger1, root.Gripper.Finger2], rotation, direction, displacement, duration)


def startCmd_Gripper(rootNode, fingers, rotation, direction, displacement, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        fingers: list
            The fingers.
        rotation, direction, displacement: float
            The elements of the commande.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

    # Definition of the elements of the animation
    def executeAnimation(fingers, rotation, direction, displacement, factor):
        if rotation is not None:
            rotateFingers(fingers, rotation)
        if direction is not None:
            translateFingers(fingers, direction)
        if displacement is not None:
            displace(fingers[0], displacement)
            displace(fingers[1], displacement)

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"fingers": fingers,
                    "rotation": rotation,
                    "direction": direction,
                    "displacement": displacement},
            duration=duration, mode="once"))


def action_to_command(action):
    """Link between Gym action (int) and SOFA command (rotation, translation,
    displacement).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).

    Returns:
    -------
        The command (rotation, direction, displacement).
    """

    rotation = None
    direction = None
    displacement = None

    if action == 0:
        direction = [0.0, 2.0, 0.0]
    elif action == 1:
        direction = [0.0, -1.0, 0]
    elif action == 2:
        direction = [1.0, 0.0, 0.0]
    elif action == 3:
        direction = [-1.0, 0.0, 0.0]
    elif action == 4:
        direction = [0.0, 0.0, 1.0]
    elif action == 5:
        direction = [0.0, 0.0, -1.0]
    elif action == 6:
        displacement = 1
    elif action == 7:
        displacement = -1

    return rotation, direction, displacement
