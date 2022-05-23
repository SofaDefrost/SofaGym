# -*- coding: utf-8 -*-

def DefaultSolver(node, iterative=True):
    '''
    Adds EulerImplicit, CGLinearSolver

    Components added:
        EulerImplicit
        CGLinearSolver
    '''
    node.addObject('EulerImplicit', name='TimeIntegrationSchema')
    if iterative:
        return node.addObject('CGLinearSolver', name='LinearSolver')

    return node.addObject('SparseLDLSolver', name='LinearSolver')

### This function is just an example on how to use the DefaultHeader function.
def createScene(rootNode):
	DefaultSolver(rootNode)
