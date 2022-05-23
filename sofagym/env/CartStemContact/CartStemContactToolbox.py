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
from splib3.animation.animate import Animation

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


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
        if kwargs["max_dist"]:
            self.max_dist = kwargs["max_dist"]
        else:
            print(">> ERROR: give a max dist for the normalization of the reward.")
            exit(1)

        self.goal = np.array(self.rootNode.GoalSetter.goalPos)
        self.sphere = self.rootNode.cartstem.Cart.MappedFrames.sphere_CollisionModel

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
        dist = float(abs(current_sphere_pos[0]-self.goal[0]))
        r = -dist  # /self.init_goal_dist
        reward = max(-self.max_dist, r)/self.max_dist
        # reward = -dist

        return reward, dist

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
        self.init_goal_dist = float(abs(current_sphere_pos[0]-self.goal[0]))

    def _getSpherePos(self):
        pos = self.sphere.MechanicalObject.position.value[0, :3]
        return pos


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


class sceneModerator(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.cartstem=None
        if kwargs["cartstem"]:
            self.cartstem = kwargs["cartstem"]

        self.contacts=None
        if kwargs["contacts"]:
            self.contacts = kwargs["contacts"]

    def getPos(self):
        return self.cartstem.getPos()

    def setPos(self, pos):
        self.cartstem.setPos(pos)


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
    contacts = rootNode.sceneModerator.contacts
    cartstem = rootNode.sceneModerator.cartstem
    factor = cartstem.max_move

    posCart = cartstem.cart.RigidBaseMO.position.value.tolist()[0][0]  # /factor
    posTip = cartstem.cart.MappedFrames.sphere_CollisionModel.MechanicalObject.position.value.tolist()[0][0]  # /factor
    posContacts = [p for p in contacts.getPos()]  # [p/factor for p in contacts.getPos()]
    goal = rootNode.GoalSetter.goalPos[0]  # /factor
    state = [posCart, posTip] + posContacts + [goal]

    return state


def getReward(rootNode):
    r, dist = rootNode.Reward.getReward()
    done = dist < 0.15
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
        if "cartstem" in kwargs:
            print(">>  Init cartstem...")
            self.cartstem = kwargs["cartstem"]
            self.cart = self.cartstem.cart
        else:
            print(">>  ERROR: No cartstem ...")
            exit(1)

        self.max_incr = self.cartstem.max_v*self.cartstem.dt
        self.set_max_move(self.cartstem.max_move)
        print(">>  Init done.")

    def set_max_move(self, new_max_move):
        self.max_move = new_max_move
        self.a, self.b = self.max_move, 0

    def _move(self, incr):
        cartMO = self.cart.RigidBaseMO
        pos = cartMO.position.value.tolist()
        if abs(pos[0][0]+incr) < self.max_move:
            pos[0][0] += incr
        cartMO.position.value = np.array(pos)

    def _normalizedAction_to_action(self, action):
        return self.a*action + self.b

    def compute_action(self, actions, nb_step):
        position_goal = self._normalizedAction_to_action(actions[0])
        current_position = self.cart.RigidBaseMO.position.value.tolist()[0][0]
        incr = (position_goal - current_position)/nb_step

        if abs(incr) > self.max_incr:
            if incr >= 0:
                incr = self.max_incr
            else:
                incr = -self.max_incr
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
    startCmd_CartStem(root, incr, duration)


def startCmd_CartStem(rootNode, incr, duration):
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
            duration=duration, mode="once"))
