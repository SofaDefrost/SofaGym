# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Mar 23 2021"

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation
from splib3.numerics import Quat
from MazeTools import Graph, dijkstra


SofaRuntime.importPlugin("SofaComponentAll")

def addRigidObject(node, filename, collisionFilename=None, position=[0,0,0,0,0,0,1], scale=[1,1,1], textureFilename='', color=[1,1,1], density=0.002, name='Object', withSolver=True, collisionGroup = 0, withCollision=True):

    if collisionFilename == None:
        collisionFilename = filename

    object = node.addChild(name)
    object.addObject('RequiredPlugin', name='SofaPlugins', pluginName='SofaRigid SofaLoader')
    object.addObject('MechanicalObject', template='Rigid3', position=position, showObject=False, showObjectScale=5)

    if withSolver:
        object.addObject('EulerImplicitSolver')
        object.addObject('CGLinearSolver', tolerance=1e-5, iterations=25, threshold = 1e-5)
        object.addObject('UncoupledConstraintCorrection')

    visu = object.addChild('Visu')
    visu.addObject('MeshOBJLoader', name='loader', filename=filename, scale3d=scale)
    visu.addObject('OglModel', src='@loader',  color=color if textureFilename =='' else '')
    visu.addObject('RigidMapping')

    object.addObject('GenerateRigidMass', name='mass', density=density, src=visu.loader.getLinkPath())
    object.mass.init()
    translation = list(object.mass.centerToOrigin.value)
    object.addObject('UniformMass', vertexMass="@mass.rigidMass")

    visu.loader.translation = translation

    if withCollision:
        collision = object.addChild('Collision')
        collision.addObject('MeshOBJLoader', name='loader', filename=collisionFilename, scale3d=scale)
        collision.addObject('MeshTopology', src='@loader')
        collision.addObject('MechanicalObject', translation=translation)
        collision.addObject('TriangleCollisionModel', group = collisionGroup)
        collision.addObject('LineCollisionModel', group = collisionGroup)
        collision.addObject('PointCollisionModel', group = collisionGroup)
        collision.addObject('RigidMapping')

    return object



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
            print("error")
            return 0.0, True
        if min([sorted_dist[0]["dist"], sorted_dist[1]["dist"]]) >= 50:
            print("ejected ball")
            return 0.0, True
        closest_points = [sorted_dist[0]["id"], sorted_dist[1]["id"]]

        new_ratio = max(self.path_length[closest_points[0]], 0)/self.path_length[-1]
        # if new_ratio < self.prev_ratio:
        #     return 0.0, None
        if new_ratio > self.prev_ratio:
            self.prev_ratio = new_ratio
            return 1.0, False
        else:
            return 0.0, False

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
                edges += [(edge[0], edge[1], np.linalg.norm(np.array(self.path_mesh.position.value[edge[0]])-np.array(self.path_mesh.position.value[edge[1]])))]

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
        new_position = self.rootNode.model.maze.Path.dofs.position.value[self.goalPos][:3]
        with self.goalMO.position.writeable() as position:
            position[0] = new_position

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


###############################################################################################################
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
    cs = 3

    goal_pos = _getGoalPos(root).tolist()
    maze_rigid_pos = root.model.rigid_maze_mo.position.value[0]
    sphere_pos = root.sphere.sphere_mo.position.value[0]

    state = [round(float(k), cs) for k in sphere_pos] + [round(float(k), cs) for k in maze_rigid_pos] + goal_pos

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

    spheres = root.sphere.sphere_mo.position.value[:3]
    goal = root.Goal.GoalMO.position.value[:3]
    if np.linalg.norm(spheres-goal) <= 10:
        print("Terminal State")
        print(np.linalg.norm(spheres-goal))
        return True, 1.0

    reward, terminal = root.Reward.getReward()
    if reward != 0.0:
        print(reward)

    return terminal, reward


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
    theta_x, theta_z = action_to_command(action)
    startCmd_Maze(root, (theta_x, theta_z), duration)


def displace(root, prev_theta_x, prev_theta_z, displacement, factor):
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

    theta_x, theta_z = displacement
    pos = root.model.rigid_maze_mo.position.value[0][:3]
    quat = Quat.createFromEuler([prev_theta_x + (theta_x - prev_theta_x) * factor,
                                 0,
                                 prev_theta_z + (theta_z - prev_theta_z) * factor])
    root.model.rigid_maze_mo.rest_position = [[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]]


def startCmd_Maze(root, displacement, duration):
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
    def executeAnimation(root, prev_theta_x, prev_theta_z, displacement, factor):
        displace(root, prev_theta_x, prev_theta_z, displacement, factor)

    prev_theta_x, _, prev_theta_z = Quat(root.model.rigid_maze_mo.position.value[0][3:]).getEulerAngles()
    # Add animation in the scene
    root.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"root": root,
                    "prev_theta_x": prev_theta_x,
                    "prev_theta_z": prev_theta_z,
                    "displacement": displacement},
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
        theta_x, theta_z = 0.2, 0.2
    elif action == 1:
        theta_x, theta_z = -0.2, 0.2
    elif action == 2:
        theta_x, theta_z = 0.2, -0.2
    elif action == 3:
        theta_x, theta_z = -0.2, -0.2
    else:
        raise NotImplementedError("Action is not in range 0 - 3")

    return theta_x, theta_z


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

    maze = root.model.maze.maze_mesh_mo.position.value.tolist()
    spheres = root.sphere.sphere_mo.position.value.tolist()

    rigid = root.model.rigid_maze_mo.position.value.tolist()

    goal = root.Goal.GoalMO.position.value.tolist()

    return [maze, spheres, rigid, goal]


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
    [maze, spheres, rigid, goal] = pos

    root.model.maze.maze_mesh_mo.position.value = np.array(maze)
    root.sphere.sphere_mo.position.value = np.array(spheres)

    root.model.rigid_maze_mo.position.value = np.array(rigid)
    root.Goal.GoalMO.position.value = np.array(goal)
