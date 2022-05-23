# -*- coding: utf-8 -*-

def FixedBox(applyTo=None, atPositions=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
             name="FixedBox",
             doVisualization=False,
             position=None,
             constraintStrength='1e12', doRecomputeDuringSimulation=False):
    """
    Constraint a set of degree of freedom to be at a fixed position.

    Args:
        applyTo (Sofa.Node): Node where the constraint will be applyied

        atPosition (vec6f): Specify min/max points of the font.

        name (str): Set the name of the FixedBox constraint.

        doVisualization (bool): Control whether or not we display the boxes.

    Structure:

    .. sourcecode:: qml

        Node : {
            name : "fixedbox",
            BoxROI,
            RestShapeSpringsFroceField
        }

    """

    c = applyTo.addChild(name)
    if position == None:
        c.addObject('BoxROI', name='BoxROI', box=atPositions, drawBoxes=doVisualization, doUpdate=doRecomputeDuringSimulation)
    else:
        c.addObject('BoxROI', position=position, name='BoxROI', box=atPositions, drawBoxes=doVisualization, doUpdate=doRecomputeDuringSimulation)

    c.addObject('RestShapeSpringsForceField', points='@BoxROI.indices', stiffness='1e12')
    return c
