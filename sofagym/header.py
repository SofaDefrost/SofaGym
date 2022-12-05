# -*- coding: utf-8 -*-
"""Header used in various scenes

Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

from splib3.animation import AnimationManagerController


def addHeader(rootNode, alarmDistance=2.0, contactDistance=0.5, tolerance=1e-6, maxIterations=500,
              gravity=[0, -981.0, 0], dt=0.03, mu='0.8', genericConstraintSolver=True):

    rootNode.addObject('RequiredPlugin', name='SofaPlugins', pluginName='SofaGeneralRigid '
                                                                        'SofaSparseSolver '
                                                                        'SofaSimpleFem '
                                                                        'SofaTopologyMapping '
                                                                        'SofaEngine '
                                                                        'SofaGeneralLinearSolver '
                                                                        'SofaRigid '
                                                                        'SofaGeneralAnimationLoop '
                                                                        'BeamAdapter '
                                                                        'SofaBoundaryCondition '
                                                                        'SofaConstraint '
                                                                        'SofaMiscCollision '
                                                                        'SofaPreconditioner '
                                                                        'SofaPython3 '
                                                                        'SofaDeformable '
                                                                        'SofaGeneralEngine '
                                                                        'SofaImplicitOdeSolver '
                                                                        'SofaMeshCollision '
                                                                        'SofaOpenglVisual '
                                                                        'CosseratPlugin '
                                                                        'SofaLoader '
                                                                        'SofaGeneralLoader')
    rootNode.addObject('BackgroundSetting', color=[1, 1, 1, 1])

    rootNode.addObject("DefaultPipeline")
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('RuleBasedContactManager', responseParams='mu='+mu, response='FrictionContactConstraint')
    rootNode.addObject('LocalMinDistance', alarmDistance=alarmDistance, contactDistance=contactDistance, angleCone=0.01)
    rootNode.addObject('FreeMotionAnimationLoop')

    if genericConstraintSolver:
        rootNode.addObject('GenericConstraintSolver', tolerance=tolerance, maxIterations=maxIterations)

    rootNode.addObject(AnimationManagerController(rootNode, name="AnimationManager"))
    rootNode.gravity = gravity
    rootNode.dt = dt


def addVisu(rootNode, config, position, direction, cutoff=250):
    source = config["source"]
    target = config["target"]
    zFar = config["zFar"]

    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels showCollisionModels')
    # showWireframe showForceFields')
    # rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")

    rootNode.addObject("LightManager")
    for i in range(len(position)):
        rootNode.addObject("SpotLight", name="SpotLight_"+str(i), position=position[i], direction=direction[i],
                           cutoff=cutoff)
    rootNode.addObject('InteractiveCamera', name='camera', position=source, lookAt=target, zFar=zFar)


# Test
def createScene(rootNode):
    addHeader(rootNode)
