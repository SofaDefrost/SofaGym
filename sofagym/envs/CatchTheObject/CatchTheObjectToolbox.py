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

        self.ball = self.rootNode.sceneModerator.ball
        self.cart = self.rootNode.sceneModerator.cart

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        pos_ball = self.ball.sphere.sphere_mo.position.value[0]
        pos_goal = self.cart.cart.Goal.GoalMO.position.value[0][:3]

        dist = np.linalg.norm(pos_ball-pos_goal)
        r = -float(dist)/self.max_dist

        under = pos_ball[2] <= 0

        return r, dist, under

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
        self.max_dist = np.linalg.norm(np.array([0, self.ball.max_high])-np.array([self.cart.max_move, 0]))


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

        self.cart = None
        if kwargs["cart"]:
            self.cart = kwargs["cart"]

        self.ball = None
        if kwargs["ball"]:
            self.ball = kwargs["ball"]

        self.gripper = None
        if kwargs["gripper"]:
            self.gripper = kwargs["gripper"]

    def getPos(self):
        p_cart = self.cart.getPos()
        p_ball = self.ball.getPos()
        p_gripper = self.gripper.getPos()

        return [p_cart, p_ball, p_gripper]

    def setPos(self, pos):
        [p_cart, p_ball, p_gripper] = pos

        self.cart.setPos(p_cart)
        self.ball.setPos(p_ball)
        self.gripper.setPos(p_gripper)


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
    ball = rootNode.sceneModerator.ball
    cart = rootNode.sceneModerator.cart
    gripper = rootNode.sceneModerator.gripper
    applyAction = rootNode.applyAction

    pos_ball = ball.sphere.sphere_mo.position.value.tolist()[0][2]/ball.max_high
    v_ball = ball.sphere.sphere_mo.velocity.value.tolist()[0][2]/175
    pos_cart = cart.cart.MechanicalObject.position.value.tolist()[0][0]/cart.max_move
    d_cart = (applyAction.direction - 0.5)*2
    pressure = gripper.cavities[0].SurfacePressureConstraint.value.value.tolist()[0]/gripper.max_pressure

    state = [pos_ball, v_ball, pos_cart, d_cart, pressure]
    return state


def getReward(rootNode):
    r, dist, under = rootNode.Reward.getReward()
    done = False
    if dist <= 2.5:
        done = True
    elif dist > rootNode.Reward.max_dist and under:
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
        self.root =  kwargs["root"]
        if "cart" in kwargs:
            print(">>  Init cart...")
            self.cart = kwargs["cart"]
        else:
            print(">>  No cart ...")
            self.cart = None

        if "gripper" in kwargs:
            print(">>  Init gripper...")
            self.gripper = kwargs["gripper"]
        else:
            print(">>  No gripper ...")
            self.gripper = None

        self.max_move = self.cart.max_move
        self.set_max_pressure(self.gripper.max_pressure)
        self.incr = 0.3
        self.direction = np.random.randint(0, 1)
        self.counter = 0
        print(">>  Init done.")

    def _moveCart(self, incr):
        cartMO = self.cart.cart.MechanicalObject
        pos = cartMO.position.value.tolist()
        if abs(pos[0][0]+incr) <= self.max_move+3:
            pos[0][0] += incr
        elif self.counter <= 25:
            self.counter +=1
        else:
            self.counter = 0
            self.direction = (self.direction+1) % 2
        cartMO.position.value = np.array(pos)

    def onAnimateBeginEvent(self, event):
        if self.direction == 1:
            self._moveCart(self.incr)
        else:
            self._moveCart(-self.incr)

    def _addPressure(self, incr):
        cavities = self.gripper.cavities
        for cavity in cavities:
            old_value = cavity.SurfacePressureConstraint.value.value[0]
            new_value = old_value + incr
            if new_value >= 0 and new_value <= self.max_pressure:
                cavity.SurfacePressureConstraint.value.value = np.array([new_value])

    def set_max_pressure(self, new_max_pressure):
        self.max_pressure = new_max_pressure
        self.a, self.b = self.max_pressure/2, self.max_pressure/2

    def _normalizedAction_to_action(self, action):
        return self.a*action + self.b

    def compute_action(self, actions, nb_step):
        pressure_goal = self._normalizedAction_to_action(actions[0])
        current_pressure = self.gripper.cavities[0].SurfacePressureConstraint.value.value[0]
        incr = (pressure_goal - current_pressure)/nb_step
        return incr

    def apply_action(self, incr):
        self._addPressure(incr)


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
    _startCmd(root, incr, duration)


def _startCmd(rootNode, incr, duration):
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
