# -*- coding: utf-8 -*-

from stlib3.splib.objectmodel import SofaPrefab, SofaObject
from stlib3.splib.scenegraph import get
from stlib3.stlib.solver import DefaultSolver


from stlib3.stlib.scene.mainheader import MainHeader
from stlib3.stlib.scene.contactheader import ContactHeader
from stlib3.stlib.scene.interaction import Interaction


def Node(parentNode, name):
    """Create a new node in the graph and attach it to a parent node."""
    return parentNode.addChild(name)

@SofaPrefab
class Scene(SofaObject):
    """Scene(SofaObject)
    Create a scene with default properties.
       Arg:
        node (Sofa.Node)     the node where the scene will be attached
        gravity (vec3f)      the gravity of the scene
        dt (float)           the dt time
        plugins (list(str))  set of plugins that are used in this scene
        repositoryPath (list(str)) set of path where to read the data from
        doDebug (bool)       activate debugging facility (to print text)
       There is method to add default solver and default contact management
       on demand.
    """
    def __init__(self, node,  gravity=[0.0, -9.81, 0.0], dt=0.01, plugins=[], repositoryPaths=[], doDebug=False):
        self.node = node
        MainHeader(node, gravity=gravity, dt=dt, plugins=plugins, repositoryPaths=repositoryPaths, doDebug=doDebug)
        self.visualstyle = self.node.VisualStyle

    def addSolver(self):
        self.solver = DefaultSolver(self.node)

    def addContact(self,  alarmDistance, contactDistance, frictionCoef=0.0):
        ContactHeader(self.node,  alarmDistance, contactDistance, frictionCoef)
