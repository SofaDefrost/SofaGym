# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Fab 3 2021"

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib.animation.animate import Animation

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

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
        else:
            print(">> ERROR: give a max dist for the normalization of the reward.")
            exit(1)

        self.goal = np.array(self.rootNode.GoalSetter.goalPos)[:2]
        self.sphere = self.rootNode.sphere

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        current_sphere_pos = self._getSpherePos()
        dist = float(np.linalg.norm(current_sphere_pos-self.goal))
        r = -dist/self.init_goal_dist

        return r, dist

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
        current_sphere_pos = self._getSpherePos()
        self.init_goal_dist = float(np.linalg.norm(current_sphere_pos-self.goal))

    def _getSpherePos(self):
        pos = self.sphere.MechanicalObject.position.value[0, :2]
        return pos


class sceneModerator(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.bubblemotion=None
        if kwargs["bubblemotion"]:
            self.bubblemotion = kwargs["bubblemotion"]

    def getPos(self):
        return self.bubblemotion.getPos()

    def setPos(self, pos):
        self.bubblemotion.setPos(pos)


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
    cavities = rootNode.sceneModerator.bubblemotion.cavities
    max_pressure = rootNode.sceneModerator.bubblemotion.max_pressure

    cavities_pos = []
    for i in range(len(cavities)):
        cavity_value = cavities[i].SurfacePressureConstraint.value.value.tolist()[0]/max_pressure
        cavities_pos.append(cavity_value)

    _sphere_pos = rootNode.sphere.MechanicalObject.position.value[0,:3].tolist()
    sphere_pos = [_sphere_pos[0]/10, _sphere_pos[1]/10, _sphere_pos[2]/5]

    _goal_pos = rootNode.GoalSetter.goalPos
    goal_pos = [_goal_pos[0]/10, _goal_pos[1]/10, _goal_pos[2]/5]

    return cavities_pos + sphere_pos + goal_pos


class goalSetter(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.goalPos = None
        if 'goalPos' in kwargs:
            self.goalPos = kwargs["goalPos"]

    def update(self):
        pass

    def set_mo_pos(self, goal):
        pass


def getReward(rootNode):
    r, dist = rootNode.Reward.getReward()

    done = False
    if dist < 0.6:
        done = True

    return done, r


def getPos(root):
    position = root.sceneModerator.getPos()
    return position


def setPos(root, pos):
    root.sceneModerator.setPos(pos)


class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        if "bubblemotion" in kwargs:
            print(">>  Init bubblemotion...")
            self.bubblemotion = kwargs["bubblemotion"]
        else:
            print(">>  ERROR: No bubblemotion ...")
            exit(1)

        self.cavities = self.bubblemotion.cavities
        self.set_max_pressure(self.bubblemotion.max_pressure)
        print(">>  Init done.")

    def set_max_pressure(self, new_max_pressure):
        self.max_pressure= new_max_pressure
        self.a, self.b = self.max_pressure/2, self.max_pressure/2

    def _move(self, cavity, incr):
        old_value = cavity.SurfacePressureConstraint.value.value[0]
        new_value = old_value + incr
        if new_value >= 0 and new_value <= self.max_pressure:
            cavity.SurfacePressureConstraint.value.value = np.array([new_value])

    def _normalizedAction_to_action(self, action):
        return self.a*action + self.b

    def compute_action(self, actions, nb_step):
        incr = []
        for i in range(len(actions)):
            goal_pressure = self._normalizedAction_to_action(actions[i])
            current_pressure = self.cavities[i].SurfacePressureConstraint.value.value[0]
            cavity_incr = (goal_pressure-current_pressure)/nb_step
            incr.append(cavity_incr)
        return incr

    def apply_action(self, incr):
        for i in range(len(incr)):
            self._move(self.cavities[i], incr[i])


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
