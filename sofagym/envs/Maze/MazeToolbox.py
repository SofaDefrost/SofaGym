# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = ("emenager", "PSC")
__contact__ = ("etienne.menager@ens-rennes.fr", "pierre.schegg@inria.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 7 2020"

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

from MazeTools import Graph, dijkstra

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
        self.goal_node = None
        if kwargs["goal_node"]:
            self.goal_node = kwargs["goal_node"]
        self.path_mesh = None
        if kwargs["path_mesh"]:
            self.path_mesh = kwargs["path_mesh"]
        self.path_mo = None
        if kwargs["path_mo"]:
            self.path_mo = kwargs["path_mo"]
        self.ball_mo = None
        if kwargs["ball_mo"]:
            self.ball_mo = kwargs["ball_mo"]

        self.start_node = 115
        self.prev_ratio = 0.0

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        pos = self.ball_mo.position.value[0]
        dist_to_path = [{'id': k, 'dist': np.linalg.norm(pos - path_point)}
                        for k, path_point in enumerate(self.path_pos)]
        sorted_dist = sorted(dist_to_path, key=lambda item: item['dist'])
        if len(sorted_dist) < 2:
            return 0.0
        closest_points = [sorted_dist[0]["id"], sorted_dist[1]["id"]]

        new_ratio = max(max(self.path_length[closest_points[0]],
                            self.path_length[closest_points[1]]), 0)/self.path_length[-1]
        if new_ratio > self.prev_ratio:
            self.prev_ratio = new_ratio
            return 1.0, None
        else:
            return new_ratio, None

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

        edges = []
        with self.path_mesh.edges.writeable() as Topoedges:
            for edge in Topoedges:
                edges += [(edge[0], edge[1], np.linalg.norm(np.array(self.path_mesh.position.value[edge[0]]) -
                                                            np.array(self.path_mesh.position.value[edge[1]])))]

        self.path_graph = Graph()
        for edge in edges:
            self.path_graph.add_edge(*edge)

        path1, path_length1 = dijkstra(self.path_graph, 30, self.goal_node)
        path2, path_length2 = dijkstra(self.path_graph, 31, self.goal_node)
        if len(path1) > len(path2):
            self.path, self.path_length = path1, path_length1
        else:
            self.path, self.path_length = path2, path_length2
        self.path_pos = []
        for point in self.path:
            self.path_pos += [self.path_mo.position.value[point][:3]]


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
        new_position = self.rootNode.Simulation.MAZE.Path.dofs.position.value[self.goalPos][:3]
        with self.goalMO.position.writeable() as position:
            position[0] = new_position

    def set_mo_pos(self, goal):
        """Modify the goal.

        Not used here.
        """
        pass


def _getGoalPos(rootNode):
    """Get XYZ position of the goal.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        The position of the goal.
    """
    return rootNode.Goal.GoalMO.position[0]


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
    cs = 3

    goalPos = _getGoalPos(rootNode).tolist()
    maze = rootNode.Simulation.MAZE.dofs.position.value[0]
    maze = [round(float(k), cs) for k in maze]

    spheres = rootNode.Simulation.spheres.dofs.position.value[0]
    spheres = [round(float(k), cs) for k in spheres]

    state = spheres + maze + goalPos

    return state


def getReward(rootNode):
    """Compute the reward using Reward.getReward().

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        done, reward

    """

    reward, cost = rootNode.Reward.getReward()

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
    num_actuator, displacement = action_to_command(action)
    actuators = [root.Modelling.Tripod.ActuatedArm0,
                 root.Modelling.Tripod.ActuatedArm1,
                 root.Modelling.Tripod.ActuatedArm2]
    actuator = actuators[num_actuator]

    startCmd_Maze(root, actuator, displacement, duration)


def displace(actuator, displacement):
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
    new_value = actuator.angleIn.value + displacement
    if new_value <= 1.5 and new_value >= -1.5:
        actuator.angleIn.value = new_value


def startCmd_Maze(rootNode, actuator, displacement, duration):
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
    def executeAnimation(actuator, displacement, factor):
        displace(actuator, displacement)

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(onUpdate=executeAnimation, params={"actuator": actuator, "displacement": displacement},
                  duration=duration, mode="once"))


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
        num_actuator, displacement = 0, 0.1
    elif action == 1:
        num_actuator, displacement = 1, 0.1
    elif action == 2:
        num_actuator, displacement = 2, 0.1
    elif action == 3:
        num_actuator, displacement = 0, -0.1
    elif action == 4:
        num_actuator, displacement = 1, -0.1
    elif action == 5:
        num_actuator, displacement = 2, -0.1
    else:
        raise NotImplementedError("Action is not in range 0, 5")

    return num_actuator, displacement


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

    root.GoalSetter.update()

    maze = root.Simulation.MAZE.collision.MechanicalObject.position.value.tolist()
    spheres = root.Simulation.spheres.dofs.position.value.tolist()

    rigid = root.Modelling.Tripod.RigidifiedStructure.RigidParts.dofs.position.value.tolist()
    deformable = root.Modelling.Tripod.RigidifiedStructure.DeformableParts.dofs.position.value.tolist()
    elastic = root.Modelling.Tripod.ElasticBody.ElasticMaterialObject.dofs.position.value.tolist()

    goal = root.Goal.GoalMO.position.value.tolist()

    return [maze, spheres, rigid, deformable, elastic, goal]


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
    [maze, spheres, rigid, deformable, elastic, goal] = pos

    root.Simulation.MAZE.collision.MechanicalObject.position.value = np.array(maze)
    root.Simulation.spheres.dofs.position.value = np.array(spheres)

    root.Modelling.Tripod.RigidifiedStructure.RigidParts.dofs.position.value = np.array(rigid)
    root.Modelling.Tripod.RigidifiedStructure.DeformableParts.dofs.position.value = np.array(deformable)
    root.Modelling.Tripod.ElasticBody.ElasticMaterialObject.dofs.position.value = np.array(elastic)

    root.Goal.GoalMO.position.value = np.array(goal)
