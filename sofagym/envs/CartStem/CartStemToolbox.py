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

        self.sphere = self.rootNode.sphere
        self.cart = self.rootNode.cartstem.Cart

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """

        sphere_pos = self._getSpherePos()
        cart_pos = self._getCartPos()
        dist = abs(sphere_pos-cart_pos)
        return 1, dist

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

    def _getSpherePos(self):
        pos = self.sphere.MechanicalObject.position.value.tolist()[0][0]
        return pos

    def _getCartPos(self):
        pos = self.cart.RigidBaseMO.position.value.tolist()[0][0]
        return pos


class sceneModerator(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.cartstem=None
        if kwargs["cartstem"]:
            self.cartstem = kwargs["cartstem"]

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
    cartstem = rootNode.sceneModerator.cartstem

    posCart = cartstem.cart.RigidBaseMO.position.value.tolist()[0][0]
    posTip = rootNode.sphere.MechanicalObject.position.value.tolist()[0][0]

    vCart = cartstem.cart.RigidBaseMO.velocity.value.tolist()[0][0]
    vTip = rootNode.sphere.MechanicalObject.velocity.value.tolist()[0][0]

    state = [posCart, posTip, vCart, vTip]
    return state


class goalSetter(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

    def update(self):
        pass

    def set_mo_pos(self, goal):
        pass


def getReward(rootNode):
    r, dist = rootNode.Reward.getReward()
    done = dist > 5

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

        self.incr = 1000

        factor_sphere = 0.05
        self.init_sphere = -factor_sphere + 2*factor_sphere*np.random.random()
        self.changed = 0

        print(">>  Init done.")

    def onAnimateBeginEvent(self, event):
        if self.changed == 2:
            pass
        elif self.changed == 0:
            with self.cartstem.stem.rateAngularDeform.rateAngularDeformMO.rest_position.writeable() as pos:
                pos[0][1] = self.init_sphere
            self.changed = 1
        elif self.changed == 1:
            with self.cartstem.stem.rateAngularDeform.rateAngularDeformMO.rest_position.writeable() as pos:
                pos[0][1] = 0
            self.changed = 2

    def _move(self, incr):
        cartForceField = self.cartstem.cart.ConstantForceField
        force = cartForceField.force.value.tolist()
        force[0] = incr
        cartForceField.force.value = np.array(force)

    def compute_action(self, actions, nb_step):
        if actions == 0:
            incr = self.incr
        else:
            incr = -self.incr

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
