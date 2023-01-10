# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "Fab 3 2021"

import numpy as np
from pyquaternion import Quaternion

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib.animation.animate import Animation

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from sofagym.utils import express_point


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


        self.baseMO = self.rootNode.stempendulum.Beam.MechanicalObject
        self.beam_len = self.rootNode.sceneModerator.stempendulum.beam_len
        self.pos_goal = np.array([0, self.beam_len])

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """

        pos = self._getPos()
        dist = np.linalg.norm(pos-self.pos_goal)
        r = -float(dist)/(2*self.beam_len)
        return r

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
        pass

    def _getPos(self):
        pos = self.baseMO.position.value.tolist()[-1][:2]
        return pos


class sceneModerator(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.stempendulum=None
        if kwargs["stempendulum"]:
            self.stempendulum = kwargs["stempendulum"]

    def getPos(self):
        return self.stempendulum.getPos()

    def setPos(self, pos):
        self.stempendulum.setPos(pos)

################################################################################

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

    stempendulum = rootNode.sceneModerator.stempendulum
    beam = stempendulum.beam
    beam_len = stempendulum.beam_len

    angBase = beam.MechanicalObject.position.value.tolist()[0][-2:]
    omegaBase = beam.MechanicalObject.velocity.value.tolist()[0][-1]/30
    posTip = [p/beam_len for p in beam.MechanicalObject.position.value.tolist()[-1][:3][:-1]]

    state = angBase + [omegaBase] + posTip

    return state


class goalSetter(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

    def update(self):
        pass

    def set_mo_pos(self, goal):
        pass


def getReward(rootNode):
    r =  rootNode.Reward.getReward()
    return False, r


def getPos(root):
    position = root.sceneModerator.getPos()
    return position

def setPos(root, pos):
    root.sceneModerator.setPos(pos)

################################################################################

class applyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root =  kwargs["root"]
        if "stempendulum" in kwargs:
            print(">>  Init stempendulum...")
            self.stempendulum = kwargs["stempendulum"]
        else:
            print(">>  ERROR: No stempendulum ...")
            exit(1)

        self.set_max_torque(self.stempendulum.max_torque)
        print(">>  Init done.")

    def set_max_torque(self, new_max_torque):
        self.max_torque = new_max_torque
        self.a, self.b =  self.max_torque , 0

    def _move(self, incr):
        baseForceField = self.stempendulum.beam.ConstantForceField
        force = baseForceField.force.value.tolist()
        if abs(force[5]+incr)<self.max_torque:
            force[5] += incr
        baseForceField.force.value = np.array(force)

    def _normalizedAction_to_action(self, action):
        return self.a*action + self.b

    def compute_action(self, actions, nb_step):
        torque_goal = self._normalizedAction_to_action(actions[0])
        current_torque = self.stempendulum.beam.ConstantForceField.force.value.tolist()[5]
        incr = (torque_goal - current_torque)/nb_step
        return incr

    def apply_action(self, incr):
        self._move(incr)


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

    #Definition of the elements of the animation
    def executeAnimation(rootNode, incr, factor):
        rootNode.applyAction.apply_action(incr)

    #Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"rootNode": rootNode,
                    "incr": incr},
            duration=duration, mode="once", realTimeClock=False))
