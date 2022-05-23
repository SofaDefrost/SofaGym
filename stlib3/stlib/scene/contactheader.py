# -*- coding: utf-8 -*-

def ContactHeader(applyTo, alarmDistance, contactDistance, frictionCoef=0.0):
    '''
    Args:
        applyTo (Sofa.Node): the node to attach the object to

        alarmDistance (float): define the distance at which the contact are integrated into
                               the detection computation.

        contactDistance (float): define the distance at which the contact response is
                                 integrated into the computation.


        frictionCoef (float, default=0.0): optional value, set to non-zero to enable
                                               a global friction in your scene.

    Structure:
        .. sourcecode:: qml

            rootNode : {
                CollisionPipeline,
                BruteForceDetection,
                RuleBasedContactManager,
                LocalMinDistance
            }
    '''
    if applyTo.getObject("DefaultPipeline", warning=False) is None:
            applyTo.addObject('DefaultPipeline')

    applyTo.addObject('BruteForceDetection')

    applyTo.addObject('RuleBasedContactManager', responseParams="mu="+str(frictionCoef),
                                                    name='Response', response='FrictionContact')
    applyTo.addObject('LocalMinDistance',
                        alarmDistance=alarmDistance, contactDistance=contactDistance,
                        angleCone=0.01)

    if applyTo.getObject("FreeMotionAnimationLoop", warning=False) is None:
            applyTo.addObject('FreeMotionAnimationLoop')

    if applyTo.getObject("GenericConstraintSolver", warning=False) is None:
            applyTo.addObject('GenericConstraintSolver', tolerance="1e-6", maxIterations="1000")

    return applyTo
