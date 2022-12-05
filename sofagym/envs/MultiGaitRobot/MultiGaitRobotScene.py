
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from MultiGaitRobotToolbox import rewardShaper, goalSetter

from splib3.animation import AnimationManagerController
import os


VISUALISATION = False

pathSceneFile = os.path.dirname(os.path.abspath(__file__))
pathMesh = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'
# Units: mm, kg, s.     Pressure in kPa = k (kg/(m.s^2)) = k (g/(mm.s^2) =  kg/(mm.s^2)

##########################################
# Reduced Basis Definition           #####
##########################################
modesRobot = pathSceneFile + "/ROM_data/modesQuadrupedWellConverged.txt"
nbModes = 63
modesPosition = [0 for i in range(nbModes)]

########################################################################
# Reduced Integration Domain for the PDMS membrane layer           #####
########################################################################
RIDMembraneFile = pathSceneFile + "/ROM_data/reducedIntegrationDomain_quadrupedMembraneWellConvergedNG005.txt"
weightsMembraneFile = pathSceneFile + "/ROM_data/weights_quadrupedMembraneWellConvergedNG005.txt"

#######################################################################
# Reduced Integration Domain for the main silicone body           #####
#######################################################################
RIDFile = pathSceneFile + '/ROM_data/reducedIntegrationDomain_quadrupedBodyWellConvergedNG003.txt'
weightsFile = pathSceneFile + '/ROM_data/weights_quadrupedBodyWellConvergedNG003.txt'

##############################################################
# Reduced Integration Domain in terms of nodes           #####
##############################################################
listActiveNodesFile = pathSceneFile + '/ROM_data/listActiveNodes_quadrupedBodyMembraneWellConvergedNG003and005.txt'

##########################################
# Reduced Order Booleans             #####
##########################################
performECSWBoolBody = True
performECSWBoolMembrane = True
performECSWBoolMappedMatrix = True
prepareECSWBool = False


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', position=[-10, 0.0, 0.0])
    return goal_mo


def createScene(rootNode, config={"source": [220, -500, 100],
                                  "target": [220, 0, 0],
                                  "goalPos": [0, 0, 0]}, mode='simu_and_visu'):

    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    rootNode.addObject('RequiredPlugin', name='SoftRobots', pluginName='SoftRobots')
    rootNode.addObject('RequiredPlugin', name='SofaPython', pluginName='SofaPython3')
    rootNode.addObject('RequiredPlugin', name='ModelOrderReduction', pluginName='ModelOrderReduction')
    rootNode.addObject('RequiredPlugin', name='SofaOpenglVisual')
    rootNode.addObject('RequiredPlugin', name="SofaSparseSolver")
    rootNode.addObject('RequiredPlugin', name="SofaConstraint")
    rootNode.addObject('RequiredPlugin', name="SofaEngine")
    rootNode.addObject('RequiredPlugin', name="SofaImplicitOdeSolver")
    rootNode.addObject('RequiredPlugin', name="SofaLoader")
    rootNode.addObject('RequiredPlugin', name="SofaMeshCollision")
    rootNode.addObject('RequiredPlugin', name="SofaGeneralLoader")
    rootNode.dt.value = 0.01

    if simu:
        rootNode.gravity.value = [0, 0, -9810]
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideBoundingCollisionModels hideForceFields '
                                                   'showInteractionForceFields hideWireframe')

    rootNode.addObject("DefaultVisualManagerLoop")
    if simu:
        rootNode.addObject('FreeMotionAnimationLoop')
        rootNode.addObject('GenericConstraintSolver', printLog=False, tolerance=1e-6, maxIterations=500)
        rootNode.addObject('DefaultPipeline', verbose=0)
        rootNode.addObject('BVHNarrowPhase')
        rootNode.addObject('BruteForceBroadPhase')
        rootNode.addObject('DefaultContactManager', response="FrictionContactConstraint", responseParams="mu=0.7")
        rootNode.addObject('LocalMinDistance', name="Proximity", alarmDistance=2.5, contactDistance=0.5, angleCone=0.01)

    if visu:
        rootNode.addObject('BackgroundSetting', color=[0, 0, 0, 1])

    solverNode = rootNode.addChild('solverNode')

    if simu:
        solverNode.addObject('EulerImplicitSolver', name='odesolver', firstOrder=False, rayleighStiffness=0.1,
                             rayleighMass=0.1, printLog=False)
        solverNode.addObject('SparseLDLSolver', name="preconditioner", template="CompressedRowSparseMatrixd")
        solverNode.addObject('GenericConstraintCorrection', solverName='preconditioner')
        solverNode.addObject('MechanicalMatrixMapperMOR', template='Vec1d,Vec1d', object1='@./reducedModel/alpha',
                             object2='@./reducedModel/alpha', nodeToParse='@./reducedModel/model',
                             performECSW=performECSWBoolMappedMatrix, listActiveNodesPath=listActiveNodesFile,
                             timeInvariantMapping1=True, timeInvariantMapping2=True, saveReducedMass=False,
                             usePrecomputedMass=False, precomputedMassPath='ROM_data/quadrupedMass_reduced63modes.txt',
                             printLog=False)

    ##########################################
    # FEM Reduced Model                      #
    ##########################################
    reducedModel = solverNode.addChild('reducedModel')
    reducedModel.addObject('MechanicalObject', template='Vec1d', name='alpha', position=modesPosition, printLog=False)
    ##########################################
    # FEM Model                              #
    ##########################################
    model = reducedModel.addChild('model')
    model.addObject('MeshVTKLoader', name='loader', filename=pathMesh+'full_quadriped_fine.vtk')
    model.addObject('TetrahedronSetTopologyContainer', src='@loader')
    model.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false', showIndicesScale=4e-5,
                    rx=0, printLog=False)
    model.addObject('ModelOrderReductionMapping', input='@../alpha', output='@./tetras', modesPath=modesRobot,
                    printLog=False, mapMatrices=0)
    model.addObject('UniformMass', name='quadrupedMass', totalMass=0.035, printLog=False)
    model.addObject('HyperReducedTetrahedronFEMForceField', template='Vec3d',
                    name='Append_HyperReducedFF_QuadrupedWellConverged_'+str(nbModes)+'modes', method='large',
                    poissonRatio=0.05,  youngModulus=70, prepareECSW=prepareECSWBool,
                    performECSW=performECSWBoolBody, nbModes=str(nbModes), modesPath=modesRobot, RIDPath=RIDFile,
                    weightsPath=weightsFile, nbTrainingSet=93, periodSaveGIE=50,printLog=False)
    model.addObject('BoxROI', name='boxROISubTopo', box=[0, 0, 0, 150, -100, 1], drawBoxes='true')
    model.addObject('BoxROI', name='membraneROISubTopo', box=[0, 0, -0.1, 150, -100, 0.1], computeTetrahedra=False,
                    drawBoxes=True)

    ##########################################
    # Sub topology                           #
    ##########################################
    modelSubTopo = model.addChild('modelSubTopo')
    modelSubTopo.addObject('TriangleSetTopologyContainer', position='@membraneROISubTopo.pointsInROI',
                           triangles="@membraneROISubTopo.trianglesInROI", name='container')
    modelSubTopo.addObject('HyperReducedTriangleFEMForceField', template='Vec3d', name='Append_subTopoFEM',
                           method='large', poissonRatio=0.49,  youngModulus=5000, prepareECSW=prepareECSWBool,
                           performECSW=performECSWBoolMembrane, nbModes=str(nbModes), modesPath=modesRobot,
                           RIDPath=RIDMembraneFile, weightsPath=weightsMembraneFile, nbTrainingSet=93,
                           periodSaveGIE=50, printLog=False)

    ##########################################
    # Constraint                             #
    ##########################################
    centerCavity = model.addChild('centerCavity')
    centerCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Center-cavity_finer.stl')
    centerCavity.addObject('MeshTopology', src='@loader', name='topo')
    centerCavity.addObject('MechanicalObject', name='centerCavity')
    centerCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                           value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                           valueType="volumeGrowth")
    centerCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    rearLeftCavity = model.addChild('rearLeftCavity')
    rearLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Left-cavity_finer.stl')
    rearLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    rearLeftCavity.addObject('MechanicalObject', name='rearLeftCavity')
    rearLeftCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                             valueType="volumeGrowth", value=0.0000, triangles='@topo.triangles', drawPressure=0,
                             drawScale=0.0002)
    rearLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')

    rearRightCavity = model.addChild('rearRightCavity')
    rearRightCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Rear-Right-cavity_finer.stl')
    rearRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    rearRightCavity.addObject('MechanicalObject', name='rearRightCavity')
    rearRightCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                              value=0.000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                              valueType="volumeGrowth")
    rearRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    frontLeftCavity = model.addChild('frontLeftCavity')
    frontLeftCavity.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_Front-Left-cavity_finer.stl')
    frontLeftCavity.addObject('MeshTopology', src='@loader', name='topo')
    frontLeftCavity.addObject('MechanicalObject', name='frontLeftCavity')
    frontLeftCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                              value=0.0000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                              valueType="volumeGrowth")
    frontLeftCavity.addObject('BarycentricMapping', name='mapping',  mapForces='false', mapMasses='false')

    frontRightCavity = model.addChild('frontRightCavity')
    frontRightCavity.addObject('MeshSTLLoader', name='loader',
                               filename=pathMesh+'quadriped_Front-Right-cavity_finer.stl')
    frontRightCavity.addObject('MeshTopology', src='@loader', name='topo')
    frontRightCavity.addObject('MechanicalObject', name='frontRightCavity')
    frontRightCavity.addObject('SurfacePressureConstraint', name="SurfacePressureConstraint", template='Vec3d',
                               value=0.0000, triangles='@topo.triangles', drawPressure=0, drawScale=0.0002,
                               valueType="volumeGrowth")
    frontRightCavity.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    if simu:
        modelCollis = model.addChild('modelCollis')
        modelCollis.addObject('MeshSTLLoader', name='loader', filename=pathMesh+'quadriped_collision.stl',
                              rotation=[0, 0, 0], translation=[0, 0, 0])
        modelCollis.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
        modelCollis.addObject('MechanicalObject', name='collisMO', template='Vec3d')
        modelCollis.addObject('TriangleCollisionModel', group=0)
        modelCollis.addObject('LineCollisionModel', group=0)
        modelCollis.addObject('PointCollisionModel', group=0)
        modelCollis.addObject('BarycentricMapping')

    ##########################################
    # Visualization                          #
    ##########################################
    if visu:
        modelVisu = model.addChild('visu')
        modelVisu.addObject('MeshSTLLoader', name='loader', filename=pathMesh+"quadriped_collision.stl")
        modelVisu.addObject('OglModel', src='@loader', template='Vec3d', color=[0.7, 0.7, 0.7, 0.6])
        modelVisu.addObject('BarycentricMapping')

    planeNode = rootNode.addChild('Plane')
    planeNode.addObject('MeshOBJLoader', name='loader', filename="mesh/floorFlat.obj", triangulate="true")
    planeNode.addObject('MeshTopology', src="@loader")
    planeNode.addObject('MechanicalObject', src="@loader", rotation=[90, 0, 0], translation=[250, 35, -1], scale=15)

    if visu:
        planeNode.addObject('OglModel', name="Visual", src="@loader", color=[1, 1, 1, 0.5], rotation=[90, 0, 0],
                            translation=[250, 35, -1], scale=15)
    if simu:
        planeNode.addObject('TriangleCollisionModel', simulated=0, moving=0, group=1)
        planeNode.addObject('LineCollisionModel', simulated=0, moving=0, group=1)
        planeNode.addObject('PointCollisionModel', simulated=0, moving=0, group=1)
        planeNode.addObject('UncoupledConstraintCorrection')
        planeNode.addObject('EulerImplicitSolver', name='odesolver')
        planeNode.addObject('CGLinearSolver', name='Solver', iterations=500, tolerance=1e-5, threshold=1e-5)

    goal_mo = add_goal_node(rootNode)
    rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goalPos=config['goalPos']))
    rootNode.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalPos=config['goalPos']))

    if simu:
        rootNode.addObject(AnimationManagerController(rootNode, name="AnimationManager"))

    if visu:
        source = config["source"]
        target = config["target"]
        rootNode.addObject("LightManager")
        spotLoc = [0, 0, 1000]
        rootNode.addObject("SpotLight", position=spotLoc, direction=[0, 0.0, -1.0])
        rootNode.addObject("InteractiveCamera", name="camera", position=source, lookAt=target, zFar=500)

    if VISUALISATION:
        print(">> Add runSofa visualisation")
        from visualisation import ApplyAction, get_config
        # path = str(pathlib.Path(__file__).parent.absolute())+"/../../../"
        config = get_config("./config_a_la_main.txt")
        config_env = config['env']
        actions = config['actions']
        scale = config_env['scale_factor']

        rootNode.addObject(ApplyAction(name="ApplyAction", root=rootNode, actions=actions, scale=scale))

    return rootNode
