# -*- coding: utf-8 -*-
from stlib3.splib.objectmodel import SofaPrefab

@SofaPrefab
class Interaction(object):
    """
    Store a list of mechanical object to interact with

    Args:

        targets ([objects])     the object to interact with and that don't have a solver

    """

    def __init__(self, parent, targets):
        self.node = parent.addChild("Interaction")
        self.node.addObject("EulerImplicitSolver")
        self.node.addObject("CGLinearSolver")
        for target in targets:
            self.node.addChild(target)
