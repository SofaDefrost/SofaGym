# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Dec 03 2021"

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation


SofaRuntime.importPlugin("Sofa.Component")


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
            self.root = kwargs["rootNode"]
        self.goal_pos = None
        if kwargs["goalPos"]:
            self.goal_pos = kwargs["goalPos"]

        self.init_dist = None
        self.prev_dist = None

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        tip = self.root.InstrumentCombined.DOFs.position[-1][:3]
        current_dist = np.linalg.norm(np.array(tip)-np.array(self.goal_pos))
        reward = max((self.prev_dist - current_dist)/self.prev_dist, 0)
        if current_dist < self.prev_dist:
            self.prev_dist = current_dist

        return min(3*reward**(1/2), 1.0), current_dist

    def update(self, goal=None):
        """Update function.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        self.goal_pos = goal
        tip = self.root.InstrumentCombined.DOFs.position[-1][:3]
        self.init_dist = np.linalg.norm(np.array(tip)-np.array(self.goal_pos))
        self.prev_dist = self.init_dist


class GoalSetter(Sofa.Core.Controller):
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

    def update(self, goal):
        """Set the position of the goal.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        self.goalPos = goal
        with self.goalMO.position.writeable() as position:
            print("update", self.goalPos)
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
    return root.Goal.GoalMO.position[0]


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
    xtips = []
    rotations = []

    for instrument in range(3):
        xtips.append(root.InstrumentCombined.m_ircontroller.xtip.value[instrument].tolist())
        rotations.append(root.InstrumentCombined.m_ircontroller.rotationInstrument.value[instrument].tolist())

    tip = root.InstrumentCombined.DOFs.position[-1][:3].tolist()

    goal_pos = _getGoalPos(root).tolist()

    state = xtips + rotations + tip + goal_pos

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

    if cost <= 3.0:
        return True, reward

    return False, reward


def get_ircontroller_state(node, instrument=0):
    """
    Get state (translation, rotation) of th Interventional Radiology Controller
    """
    return [float(node.m_ircontroller.xtip.value[instrument]),
            float(node.m_ircontroller.rotationInstrument.value[instrument])]


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
    scale = int(duration/0.01 + 1)
    controlled_instrument, cmd_translation, cmd_rotation = action_to_command(action, scale)
    source = get_ircontroller_state(root.InstrumentCombined, instrument=controlled_instrument)
    target_translation = source[0] + cmd_translation
    target = [target_translation if target_translation > 0 else 0.1, source[1] + cmd_rotation]
    start_cmd(root, root.InstrumentCombined, source, target, duration, controlled_instrument)

def start_cmd(rootNode, IRC_node, source, target, duration, instrument=0):
    def execute_animation(controller, anim_source, anim_target, factor, anim_instrument):
        """
        Execute animation on the IRC to go from source to target
        """
        with controller.xtip.writeable() as xtip:
            xtip[anim_instrument] = anim_source[0] + (anim_target[0] - anim_source[0]) * factor
        if anim_instrument == 0:
            with controller.rotationInstrument.writeable() as rotation:
                rotation[0] = anim_source[1] + (anim_target[1] - anim_source[1]) * factor

    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=execute_animation,
            params={"controller": IRC_node.m_ircontroller,
                    "anim_source": source,
                    "anim_target": target,
                    "anim_instrument": instrument},
            duration=duration, mode="once"))

    return


def action_to_command(action, scale):
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
        controlled_instrument = 0
        cmd_translation = 2.0 * scale / 2.0
        cmd_rotation = 0.0
    elif action == 1:
        controlled_instrument = 0
        cmd_translation = 0.0
        cmd_rotation = 1/15 * scale / 2.0
    elif action == 2:
        controlled_instrument = 0
        cmd_translation = 0.0
        cmd_rotation = -1/15 * scale / 2.0
    elif action == 3:
        controlled_instrument = 0
        cmd_translation = -0.7 * scale / 2.0
        cmd_rotation = 0.0

    elif action == 4:
        controlled_instrument = 1
        cmd_translation = 2.0 * scale / 2.5
        cmd_rotation = 0.0
    elif action == 5:
        controlled_instrument = 1
        cmd_translation = 0.0
        cmd_rotation = 1/15 * scale / 2.5
    elif action == 6:
        controlled_instrument = 1
        cmd_translation = 0.0
        cmd_rotation = -1/15 * scale / 2.5
    elif action == 7:
        controlled_instrument = 1
        cmd_translation = -0.7 * scale / 2.5
        cmd_rotation = 0.0

    elif action == 8:
        controlled_instrument = 2
        cmd_translation = 2.0 * scale / 3.0
        cmd_rotation = 0.0
    elif action == 9:
        controlled_instrument = 2
        cmd_translation = 0.0
        cmd_rotation = 1/15 * scale / 3.0
    elif action == 10:
        controlled_instrument = 2
        cmd_translation = 0.0
        cmd_rotation = -1/15 * scale / 3.0
    elif action == 11:
        controlled_instrument = 2
        cmd_translation = -0.7 * scale / 3.0
        cmd_rotation = 0.0

    else:
        raise NotImplementedError("Action is not in range 0 - 11")

    return controlled_instrument, cmd_translation, cmd_rotation


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
    cath_xtip = root.InstrumentCombined.m_ircontroller.xtip.value[0].tolist()
    cath_rotation = root.InstrumentCombined.m_ircontroller.rotationInstrument.value[0].tolist()
    guide_xtip = root.InstrumentCombined.m_ircontroller.xtip.value[1].tolist()
    guide_rotation = root.InstrumentCombined.m_ircontroller.rotationInstrument.value[1].tolist()    
    coils_xtip = root.InstrumentCombined.m_ircontroller.xtip.value[2].tolist()
    coils_rotation = root.InstrumentCombined.m_ircontroller.rotationInstrument.value[2].tolist()

    tip = root.InstrumentCombined.DOFs.position.value.tolist()
    collis = root.InstrumentCombined.Collis.CollisionDOFs.position.value.tolist()
    
    return [cath_xtip, cath_rotation, guide_xtip, guide_rotation, coils_xtip, coils_rotation, tip, collis]


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
    cath_xtip, cath_rotation, guide_xtip, guide_rotation, coils_xtip, coils_rotation, tip, collis = pos
    
    controller = root.InstrumentCombined.m_ircontroller
    with controller.xtip.writeable() as xtip:
        xtip[0] = np.array(cath_xtip)
        xtip[1] = np.array(guide_xtip)
        xtip[2] = np.array(coils_xtip)
    
    with controller.rotationInstrument.writeable() as rotation:
        rotation[0] = np.array(cath_rotation)
        rotation[1] = np.array(guide_rotation)
        rotation[2] = np.array(coils_rotation)

    root.InstrumentCombined.DOFs.position.value = np.array(tip)
