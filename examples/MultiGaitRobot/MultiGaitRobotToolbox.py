# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "emenager"
__contact__ = "etienne.menager@ens-rennes.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

import numpy as np
import sys

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

SofaRuntime.importPlugin("SofaComponentAll")

CENTER_PRESSUR = 2000  # 3500
LEG_PRESSUR = 1500  # 2000


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

        # With position
        self.penalty = 0
        self.pred = 0

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        robotFront = self._computeFront()
        current_dist = np.sum(np.abs(robotFront-self.goal_pos))

        # With velocity
        # return self._computeVelocity(), current_dist

        # With position: init_dist is not a distance but the initial x position
        # comment the next line during test.
        # self.penalty += 0
        pos = self._computePos()

        reward = max(0, pos[0] - self.pred)
        self.pred = pos[0]

        # print("ABSOLUTE POS:" ,pos[0] - self.init_dist)
        # reward = (pos[0] - self.init_dist)

        # print(">>> Distance to positive x:", pos[0] - self.init_dist)
        return reward, current_dist  # pos[0] - self.init_dist, current_dist  # (-10 + pos[0] - self.init_dist)/10,current_dist

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

        # robotFront = self._computeFront()

        # With velocity
        # self.init_dist = np.sqrt(np.sum((robotFront-self.goal_pos)**2))

        # With position
        pos = self._computePos()
        self.init_dist = pos[0]
        self.pred = pos[0]

    def _computeFront(self):
        """Compute the position of the front of the robot.

        Parameters:
        ----------
            None.

        Return:
        ------
            The position of the front.
        """
        front_left = self.rootNode.solverNode.reducedModel.model.centerCavity.centerCavity.position.value[426]
        front_right = self.rootNode.solverNode.reducedModel.model.centerCavity.centerCavity.position.value[622]

        return (front_left+front_right)/2

    def _computeVelocity(self):
        """Compute the  average velocity of tot_points points of the robot.

        Parameters:
        ----------
            None.

        Return:
        ------
            The average velocity.
        """
        list_point_legs = [100*i for i in range(29)]
        list_point_cent = [100*i for i in range(65)]
        tot_points = 4*len(list_point_legs) + len(list_point_cent)

        node = self.rootNode.solverNode.reducedModel.model
        points_RLC = np.sum(node.rearLeftCavity.rearLeftCavity.velocity.value[list_point_legs], axis=0)
        points_GLC = np.sum(node.rearRightCavity.rearRightCavity.velocity.value[list_point_legs], axis=0)
        points_RFC = np.sum(node.frontLeftCavity.frontLeftCavity.velocity.value[list_point_legs], axis=0)
        points_GFC = np.sum(node.frontRightCavity.frontRightCavity.velocity.value[list_point_legs], axis=0)
        points_cent = np.sum(node.centerCavity.centerCavity.velocity.value[list_point_cent], axis=0)

        vel = (points_RLC + points_GLC + points_RFC + points_GFC + points_cent)/tot_points

        return vel[0]

    def _computePos(self):
        """Compute the  average position of tot_points points of the robot.

        Parameters:
        ----------
            None.

        Return:
        ------
            The average position.
        """
        # list_point_legs = [100*i for i in range(29)]
        # list_point_cent = [100*i for i in range(65)]

        list_point_legs = [i for i in range(29*100)]
        list_point_cent = [i for i in range(65*100)]
        tot_points = 4*len(list_point_legs) + len(list_point_cent)

        node = self.rootNode.solverNode.reducedModel.model
        points_RLC = np.sum(node.rearLeftCavity.rearLeftCavity.position.value[list_point_legs], axis=0)
        points_GLC = np.sum(node.rearRightCavity.rearRightCavity.position.value[list_point_legs], axis=0)
        points_RFC = np.sum(node.frontLeftCavity.frontLeftCavity.position.value[list_point_legs], axis=0)
        points_GFC = np.sum(node.frontRightCavity.frontRightCavity.position.value[list_point_legs], axis=0)
        points_cent = np.sum(node.centerCavity.centerCavity.position.value[list_point_cent], axis=0)

        pos = (points_RLC + points_GLC + points_RFC + points_GFC + points_cent)/tot_points

        return pos


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

    """
    cs = 3
    list_point_legs = [100*i for i in range(29)]
    list_point_cent = [100*i for i in range(65)]

    nb_points = 4*len(list_point_legs)+len(list_point_cent)

    node = rootNode.solverNode.reducedModel.model
    points_RLC = np.sum(node.rearLeftCavity.rearLeftCavity.position.value[list_point_legs], axis = 0)
    points_GLC = np.sum(node.rearRightCavity.rearRightCavity.position.value[list_point_legs], axis = 0)
    points_RFC = np.sum(node.frontLeftCavity.frontLeftCavity.position.value[list_point_legs], axis = 0)
    points_GFC = np.sum(node.frontRightCavity.frontRightCavity.position.value[list_point_legs], axis = 0)
    points_cent = np.sum(node.centerCavity.centerCavity.position.value[list_point_cent], axis = 0)
    points = (points_RLC + points_GLC + points_RFC + points_GFC + points_cent)/nb_points

    points = [round(float(p), cs) for p in points]

    """
    alpha = root.solverNode.reducedModel.alpha.position.value
    n_alpha = np.linalg.norm(alpha)
    if n_alpha != 0:
        alpha = alpha / n_alpha

    alpha = [float(a) for a in alpha]

    state = alpha[:32]
    """
    node = rootNode.solverNode.reducedModel.model
    p1 = float(node.centerCavity.SurfacePressureConstraint.value.value[0])
    p2 = float(node.rearLeftCavity.SurfacePressureConstraint.value.value[0])
    p3 = float(node.rearRightCavity.SurfacePressureConstraint.value.value[0])
    p4 = float(node.frontLeftCavity.SurfacePressureConstraint.value.value[0])
    p5 = float(node.frontRightCavity.SurfacePressureConstraint.value.value[0])

    state = [p1,p2,p3,p4,p5]
    """
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
    part, pressure = action_to_command(action, root)
    startCmd_MultiGaitRobot(root, part, pressure, duration)


def changePressure(root, num_part, pressure, nb_step):
    """Change the value of the pressure in the part.

    Parameters:
    ----------
        root: <Sofa>
            The root node.
        num_part: int
            The number of the part.
        pressure: int
            The increment of the pressure.
        nb_step: int
            In the continue case, time to pass from one pressur to another.

    Returns:
    -------
        None.

    """
    """
    #discrete
    node = root.solverNode.reducedModel.model
    if num_part == 0:
        part = node.centerCavity
    elif num_part == 1:
        part = node.rearLeftCavity
    elif num_part == 2:
        part = node.rearRightCavity
    elif num_part == 3:
        part = node.frontLeftCavity
    elif num_part == 4:
        part = node.frontRightCavity
    else:
        print("Action not found")
        sys.exit(1)

    pressureValue = part.SurfacePressureConstraint.value.value + pressure

    if pressureValue >= 0:
        if num_part == 0 and pressureValue <= CENTER_PRESSUR:
            part.SurfacePressureConstraint.value.value = pressureValue
        elif num_part!=0 and pressureValue <= LEG_PRESSUR:
            part.SurfacePressureConstraint.value.value = pressureValue
    """

    # # Continue
    # node = root.solverNode.reducedModel.model
    # node.centerCavity.SurfacePressureConstraint.value.value = \
    #     node.centerCavity.SurfacePressureConstraint.value.value + pressure[0]/nb_step
    # node.rearLeftCavity.SurfacePressureConstraint.value.value = \
    #     node.rearLeftCavity.SurfacePressureConstraint.value.value + pressure[1]/nb_step
    # node.rearRightCavity.SurfacePressureConstraint.value.value = \
    #     node.rearRightCavity.SurfacePressureConstraint.value.value + pressure[2]/nb_step
    # node.frontLeftCavity.SurfacePressureConstraint.value.value = \
    #     node.frontLeftCavity.SurfacePressureConstraint.value.value + pressure[3]/nb_step
    # node.frontRightCavity.SurfacePressureConstraint.value.value = \
    #     node.frontRightCavity.SurfacePressureConstraint.value.value + pressure[4]/nb_step

    # Symetric - Continue
    # node = root.solverNode.reducedModel.model
    # node.centerCavity.SurfacePressureConstraint.value.value = \
    #     node.centerCavity.SurfacePressureConstraint.value.value + pressure[0]/nb_step
    # node.rearLeftCavity.SurfacePressureConstraint.value.value = \
    #     node.rearLeftCavity.SurfacePressureConstraint.value.value + pressure[1]/nb_step
    # node.rearRightCavity.SurfacePressureConstraint.value.value = \
    #     node.rearRightCavity.SurfacePressureConstraint.value.value + pressure[1]/nb_step
    # node.frontLeftCavity.SurfacePressureConstraint.value.value = \
    #     node.frontLeftCavity.SurfacePressureConstraint.value.value + pressure[2]/nb_step
    # node.frontRightCavity.SurfacePressureConstraint.value.value = \
    #     node.frontRightCavity.SurfacePressureConstraint.value.value + pressure[2]/nb_step

    # Discrete with max or min
    node = root.solverNode.reducedModel.model
    if num_part == 0:
        part = node.centerCavity
    elif num_part == 1:
        part1 = node.rearLeftCavity
        part2 = node.rearRightCavity
    elif num_part == 2:
        part1 = node.frontLeftCavity
        part2 = node.frontRightCavity
    else:
        print("Action not found")
        sys.exit(1)

    factor = 4
    if num_part == 0:
        pressureValue = part.SurfacePressureConstraint.value.value + factor*pressure/nb_step
        if pressureValue >= 0 and pressureValue <= CENTER_PRESSUR:
            part.SurfacePressureConstraint.value.value = pressureValue
    else:
        pressureValue1 = part1.SurfacePressureConstraint.value.value + factor*pressure/nb_step
        pressureValue2 = part2.SurfacePressureConstraint.value.value + factor*pressure/nb_step
        if pressureValue1 >= 0 and pressureValue1 <= LEG_PRESSUR and pressureValue2 >= 0 and \
                pressureValue2 <= LEG_PRESSUR:
            part1.SurfacePressureConstraint.value.value = pressureValue1
            part2.SurfacePressureConstraint.value.value = pressureValue2

    # print(part.SurfacePressureConstraint.value.value)


def startCmd_MultiGaitRobot(rootNode, num_part, pressure, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        cable: <MechanicalObject>
            The mechanical object of the cable to move.
        displacement: float
            The elements of the commande.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

    # Definition of the elements of the animation
    def executeAnimation(rootNode, num_part, pressure, nb_step, factor):
        changePressure(rootNode, num_part, pressure, nb_step)

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"rootNode": rootNode,
                    "num_part": num_part,
                    "pressure":pressure,
                    "nb_step": duration/0.01 + 1},
            duration=duration, mode="once"))


def action_to_command(action, root):
    """Link between Gym action (int) and SOFA command (displacement of cables).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).
        root:
            The root of the scene.

    Returns:
    -------
        The command (number of the cabl and its displacement).
    """
    """
    #discrete each legs
    if   action == 0:
        part, pressure = 0, 50
    elif action == 1:
        part, pressure = 1, 50
    elif action == 2:
        part, pressure = 2, 50
    elif action == 3:
        part, pressure = 3, 50
    elif action == 4:
        part, pressure = 4, 50
    elif action == 5:
        part, pressure = 0, -50
    elif action == 6:
        part, pressure = 1, -50
    elif action == 7:
        part, pressure = 2, -50
    if   action == 8:
        part, pressure = 3, -50
    elif action == 9:
        part, pressure = 4, -50

    return part, pressure
    """

    # # Continue
    # pressure_leg, pressure_center = LEG_PRESSUR, CENTER_PRESSUR
    # node = root.solverNode.reducedModel.model
    # a_center, b_center = pressure_center/2, pressure_center/2
    # a_leg, b_leg = pressure_leg/2, pressure_leg/2
    #
    # goal1 = a_center*action[0]+ b_center
    # goal2 = a_leg*action[1] + b_leg
    # goal3 = a_leg*action[2] + b_leg
    # goal4 = a_leg*action[3] + b_leg
    # goal5 = a_leg*action[4] + b_leg
    #
    # old1 = float(node.centerCavity.SurfacePressureConstraint.value.value[0])
    # old2 = float(node.rearLeftCavity.SurfacePressureConstraint.value.value[0])
    # old3 = float(node.rearRightCavity.SurfacePressureConstraint.value.value[0])
    # old4 = float(node.frontLeftCavity.SurfacePressureConstraint.value.value[0])
    # old5 = float(node.frontRightCavity.SurfacePressureConstraint.value.value[0])
    #
    # incr = [goal1 - old1, goal2-old2, goal3-old3, goal4 - old4, goal5 - old5]
    # return None, incr

    # # Symetric - Continue
    # pressure_leg, pressure_center = LEG_PRESSUR, CENTER_PRESSUR
    # node = root.solverNode.reducedModel.model
    # a_center, b_center = pressure_center/2, pressure_center/2
    # a_leg, b_leg = pressure_leg/2, pressure_leg/2
    #
    # goal1 = a_center*action[0]+ b_center
    # goal2 = a_leg*action[1] + b_leg
    # goal3 = a_leg*action[2] + b_leg
    #
    # old1 = float(node.centerCavity.SurfacePressureConstraint.value.value[0])
    # old2 = float(node.rearLeftCavity.SurfacePressureConstraint.value.value[0])
    # old3 = float(node.frontLeftCavity.SurfacePressureConstraint.value.value[0])
    #
    # incr = [goal1 - old1, goal2-old2, goal3-old3]
    # return None, incr

    # discrete with max or min
    if action == 0:
        part, pressure = 0, CENTER_PRESSUR
    elif action == 1:
        part, pressure = 1, LEG_PRESSUR
    elif action == 2:
        part, pressure = 2, LEG_PRESSUR
    elif action == 3:
        part, pressure = 0, -CENTER_PRESSUR
    elif action == 4:
        part, pressure = 1, -LEG_PRESSUR
    elif action == 5:
        part, pressure = 2, -LEG_PRESSUR

    return part, pressure


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
    node = root.solverNode.reducedModel.model
    points_RLC = node.rearLeftCavity.rearLeftCavity.position.value.tolist()
    points_GLC = node.rearRightCavity.rearRightCavity.position.value.tolist()
    points_RFC = node.frontLeftCavity.frontLeftCavity.position.value.tolist()
    points_GFC = node.frontRightCavity.frontRightCavity.position.value.tolist()
    points_cent = node.centerCavity.centerCavity.position.value.tolist()
    model_pos = node.tetras.position.value.tolist()
    return [points_RLC, points_GLC, points_RFC, points_GFC, points_cent, model_pos]


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
    node = root.solverNode.reducedModel.model
    [points_RLC, points_GLC, points_RFC, points_GFC, points_cent, model_pos] = pos
    node.rearLeftCavity.rearLeftCavity.position.value = np.array(points_RLC)
    node.rearRightCavity.rearRightCavity.position.value = np.array(points_GLC)
    node.frontLeftCavity.frontLeftCavity.position.value = np.array(points_RFC)
    node.frontRightCavity.frontRightCavity.position.value = np.array(points_GFC)
    node.centerCavity.centerCavity.position.value = np.array(points_cent)
    node.tetras.position.value = np.array(model_pos)
