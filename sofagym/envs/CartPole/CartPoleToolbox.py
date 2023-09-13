import math
import pathlib
import sys

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

SofaRuntime.importPlugin("Sofa.Component")


class StateInitializer(Sofa.Core.Controller):
    """Initialize the states.

    Methods:
    -------
        __init__: Initialization of all arguments.
        init_state: Randomly initialize the environment state.

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
        if kwargs["pole_length"]:
            self.pole_length = kwargs["pole_length"]
        if kwargs["init_states"] is not None:
            self.init_states = kwargs["init_states"]
        else:
            print(">> ERROR: no inital states given.")
            exit(1)

        self.cart = self.rootNode.Modeling.Cart
        self.pole = self.rootNode.Modeling.Pole

    def init_state(self):
        """Randomly initialize the environment state.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.

        """
        cart_pos, cart_vel, pole_theta, pole_theta_dot = self.init_states

        with self.cart.MechanicalObject.position.writeable() as position:
            position[0][0] = cart_pos

        with self.cart.MechanicalObject.velocity.writeable() as velocity:
            velocity[0][0] = cart_vel

        with self.pole.MechanicalObject.velocity.writeable() as theta_dot:
            theta_dot[0][5] = pole_theta_dot

        sin_theta = math.sin(pole_theta)
        cos_theta = math.cos(pole_theta)
        x_pos = sin_theta * self.pole_length
        y_pos = cos_theta * self.pole_length

        with self.pole.MechanicalObject.position.writeable() as position:
            position[0][0] = x_pos
            position[0][1] = y_pos


class RewardShaper(Sofa.Core.Controller):
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
        if kwargs["max_angle"]:
            self.max_angle = kwargs["max_angle"]
        else:
            print(">> ERROR: give a max angle for the normalization of the reward.")
            exit(1)
        if kwargs["pole_length"]:
            self.pole_length = kwargs["pole_length"]

        self.cart = self.rootNode.Modeling.Cart
        self.pole = self.rootNode.Modeling.Pole

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        cart_pos = self._getCartPos()
        pole_theta, pole_theta_dot = self.calculatePoleTheta(cart_pos)
        
        return 1, pole_theta, self.max_angle

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

    def _getPolePos(self):
        pos = self.pole.MechanicalObject.position.value.tolist()[0]
        return pos[0], pos[1]

    def _getCartPos(self):
        pos = self.cart.MechanicalObject.position.value.tolist()[0][0]
        return pos
    
    def calculatePoleTheta(self, cart_pos):
        x_pos, y_pos = self._getPolePos()
        sin_theta = (y_pos/self.pole_length)
        theta = abs((90*math.pi/180) - math.asin(sin_theta))
        
        if x_pos < cart_pos:
            theta = -theta

        theta_dot = self.pole.MechanicalObject.velocity.value.tolist()[0][5]
        
        return theta, theta_dot


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
    cart = rootNode.Modeling.Cart

    cart_pos = cart.MechanicalObject.position.value.tolist()[0][0]
    cart_vel = cart.MechanicalObject.velocity.value.tolist()[0][0]

    pole_theta, pole_theta_dot = rootNode.Reward.calculatePoleTheta(cart_pos)

    state = [cart_pos, cart_vel, pole_theta, pole_theta_dot]

    return state


class GoalSetter(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

    def update(self):
        pass

    def set_mo_pos(self, goal):
        pass


def getReward(rootNode):
    reward, theta, max_angle = rootNode.Reward.getReward()
    done = (theta > max_angle) or (theta < -max_angle)

    return done, reward


def getPos(root):
    cart_pos = root.Modeling.Cart.MechanicalObject.position.value[:].tolist()
    pole_pos = root.Modeling.Pole.MechanicalObject.position.value[:].tolist()
    return [cart_pos, pole_pos]


def setPos(root, pos):
    cart_pos, pole_pos = pos
    root.Modeling.Cart.MechanicalObject.position.value = np.array(cart_pos)
    root.Modeling.Pole.MechanicalObject.position.value = np.array(pole_pos)


class ApplyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        self.cart = self.root.Modeling.Cart

        self.force_scale = 5
        self.incr = 1000 * self.force_scale

    def _move(self, incr):
        cartForceField = self.cart.CartForce
        force = cartForceField.force.value.tolist()
        force[0] = incr
        cartForceField.force.value = np.array(force)

    def compute_action(self, actions):
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

    incr = root.ApplyAction.compute_action(actions)
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
    startCmd_CartPole(root, incr, duration)


def startCmd_CartPole(rootNode, incr, duration):
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
        rootNode.ApplyAction.apply_action(incr)

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"rootNode": rootNode,
                    "incr": incr},
            duration=duration, mode="once"))
