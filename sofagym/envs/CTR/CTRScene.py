# -*- coding: utf-8 -*-
"""Toolbox: compute reward, create scene, ...
"""

__authors__ = "PSC"
__contact__ = "pierre.schegg@robocath.com"
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Robocath, CNRS, Inria"
__date__ = "Nov 28 2021"


import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from os.path import dirname, abspath
import SofaRuntime
from splib.animation import AnimationManagerController

from CTRToolbox import RewardShaper, GoalSetter

# Register all the common component in the factory.
SofaRuntime.importPlugin("SofaComponentAll")

path = dirname(abspath(__file__)) + '/'


def add_plugins(root):
    root.addObject('RequiredPlugin', pluginName='BeamAdapter')
    root.addObject('RequiredPlugin', name='SofaOpenglVisual')

    root.addObject('RequiredPlugin', name='SofaBoundaryCondition')
    root.addObject('RequiredPlugin', name='SofaConstraint')
    root.addObject('RequiredPlugin', name='SofaDeformable')
    root.addObject('RequiredPlugin', name='SofaGeneralLinearSolver')
    root.addObject('RequiredPlugin', name='SofaImplicitOdeSolver')
    root.addObject('RequiredPlugin', name='SofaLoader')
    root.addObject('RequiredPlugin', name='SofaMeshCollision')
    root.addObject('RequiredPlugin', name='SofaTopologyMapping')

    return root


def add_visuals_and_solvers(root, config, mode_simu=True, mode_visu=True):
    if mode_visu:
        source = config["source"]
        target = config["target"]

        root.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideMappings hideForceFields hideWireframe')
        # root.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")

        root.addObject("LightManager")
        root.addObject("SpotLight", position=source, direction=[target[i] - source[i] for i in range(len(source))])
        root.addObject('InteractiveCamera', name='camera', position=source, lookAt=target, zFar=5000)
        # root.addObject('BackgroundSetting', color=[1, 1, 1, 1])
    root.addObject('FreeMotionAnimationLoop')
    lcp_solver = root.addObject('LCPConstraintSolver', mu=0.1, tolerance=3e-4, maxIt=10000, build_lcp=False)

    if mode_simu:
        root.addObject('DefaultPipeline', draw=False, depth=6, verbose=False)
        # root.addObject('BruteForceBroadPhase')
        # root.addObject('BVHNarrowPhase')

        root.addObject('GenericConstraintSolver', maxIterations=500, tolerance=1e-8)
        root.addObject('BruteForceDetection', name="N2")
        root.addObject('DefaultContactManager', name='Response', response="FrictionContact")
        root.addObject('LocalMinDistance', contactDistance=0.1, alarmDistance=1.0, name='localmindistance',
                       angleCone=0.2)

        root.addObject(AnimationManagerController(name="AnimationManager"))

    return root, lcp_solver


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=2.0,
                             showColor=[0, 1, 0, 0.5], position=[0.0, 0.0, 0.0])
    return goal_mo


def add_instrument_topology(root, name, i):
    # Topo Guide
    topology_node = root.addChild('topoLines_'+name)
    topology_node.addObject('WireRestShape', name=name+'RestShape', straightLength=475+i*5, length=500, numEdges=200,
                            youngModulus=50000, spireDiameter=25-i*5, numEdgesCollis=[50, 10], printLog=False,
                            template='Rigid3d', spireHeight=0.0, densityOfBeams=[30, 5],
                            youngModulusExtremity=50000*(i+1))
    topology_node.addObject('MechanicalObject', name='dofTopo2', template='Rigid3d')
    topology_node.addObject('EdgeSetTopologyContainer', name='meshLines'+name)
    topology_node.addObject('EdgeSetTopologyModifier', name='Modifier')
    topology_node.addObject('EdgeSetGeometryAlgorithms', name='GeomAlgo', template='Rigid3d')


def add_instruments_combined(root, instrument_list, mode_simu=True):
    # Instruments
    instrument_combined_node = root.addChild('InstrumentCombined')
    instrument_combined_node.addObject('VisualStyle', displayFlags='hideWireframe')
    # if mode_simu:
    instrument_combined_node.addObject('EulerImplicitSolver', rayleighStiffness=0.2, printLog=False, rayleighMass=0.1)
    instrument_combined_node.addObject('BTDLinearSolver', verification=False, subpartSolve=False, verbose=False)
    instrument_combined_node.addObject('RegularGridTopology', name='meshLinesCombined', zmax=1, zmin=1, nx=60, ny=1,
                                       nz=1, xmax=1.0, xmin=0.0, ymin=0, ymax=0)
    instrument_combined_node.addObject('MechanicalObject', showIndices=False, name='DOFs', template='Rigid3d', ry=-90)

    for instrument in instrument_list:
        instrument_combined_node.addObject('WireBeamInterpolation',
                                           WireRestShape='@../topoLines_'+instrument+'/'+instrument+'RestShape',
                                           radius=0.15, printLog=False, name='Interpol'+instrument,
                                           edgeList=[k for k in range(60)])
        instrument_combined_node.addObject('AdaptiveBeamForceFieldAndMass', massDensity=0.00000155,
                                           name=instrument+'ForceField', interpolation='@Interpol'+instrument)

    instrument_combined_node.addObject('InterventionalRadiologyController', xtip=[1, 0, 0], name='m_ircontroller',
                                       instruments=['Interpol'+instrument for instrument in instrument_list], step=0.5,
                                       printLog=False, listening=True, template='Rigid3d',
                                       startingPos=[0, 0, 0, 0, 0, 0, 1], rotationInstrument=[0, 0, 0], speed=0,
                                       controlledInstrument=0)
    if mode_simu:
        instrument_combined_node.addObject('LinearSolverConstraintCorrection', wire_optimization='true', printLog=False)
    instrument_combined_node.addObject('FixedConstraint', indices=0, name='FixedConstraint')
    instrument_combined_node.addObject('RestShapeSpringsForceField', points='@m_ircontroller.indexFirstNode',
                                       angularStiffness=1e0, stiffness=1e0)

    if mode_simu:
        # Collision model
        Collis = instrument_combined_node.addChild('Collis')
        Collis.addObject('EdgeSetTopologyContainer', name='collisEdgeSet')
        Collis.addObject('EdgeSetTopologyModifier', name='colliseEdgeModifier')
        Collis.addObject('MechanicalObject', name='CollisionDOFs')
        Collis.addObject('MultiAdaptiveBeamMapping', controller='../m_ircontroller',
                         useCurvAbs=True, printLog=False, name='collisMap')
        Collis.addObject('LineCollisionModel', proximity=0.0, group=1)
        Collis.addObject('PointCollisionModel', proximity=0.0, group=1)

    return instrument_combined_node


def add_instrument_visual(instrument_combined_node, instrument_name, color, i):
    visu_node = instrument_combined_node.addChild('Visu_'+instrument_name)
    visu_node.addObject('MechanicalObject', name='Quads')
    visu_node.addObject('QuadSetTopologyContainer', name='Container'+instrument_name)
    visu_node.addObject('QuadSetTopologyModifier', name='Modifier')
    visu_node.addObject('QuadSetGeometryAlgorithms', name='GeomAlgo', template='Vec3d')
    visu_node.addObject('Edge2QuadTopologicalMapping', radius=0.5*(i+1), listening='true',
                        input='@../../topoLines_'+instrument_name+'/meshLines'+instrument_name,
                        nbPointsOnEachCircle='10', flipNormals='true', output='@Container'+instrument_name)
    visu_node.addObject('AdaptiveBeamMapping', interpolation='@../Interpol'+instrument_name,
                        name='visuMap'+instrument_name, output='@Quads', isMechanical=False,
                        input='@../DOFs', useCurvAbs=True, printLog=False)

    # Ogl model
    visu_ogl = visu_node.addChild('VisuOgl')
    visu_ogl.addObject('OglModel', color=color, quads='@../Container'+instrument_name+'.quads', name='Visual')
    visu_ogl.addObject('IdentityMapping', input='@../Quads', output='@Visual')


def add_geometry(root, config, mode_simu=True, mode_visu=True):
    geometry = root.addChild('Vessels')
    geometry.addObject('MeshObjLoader', filename=path+config["mesh"], flipNormals=True, triangulate=True,
                       name='meshLoader', scale=config["scale"], translation=config["translation"],
                       rotation=config["rotation"], printLog=False)
    geometry.addObject('MeshTopology', position='@meshLoader.position', triangles='@meshLoader.triangles')
    mo = geometry.addObject('MechanicalObject', name='DOFs1')
    if mode_simu:
        geometry.addObject('TriangleCollisionModel', moving=False, simulated=False)
        geometry.addObject('LineCollisionModel', moving=False, simulated=False)
        geometry.addObject('PointCollisionModel', moving=False, simulated=False)
    if mode_visu:
        geometry.addObject('OglModel', color=[0.8, 0.8, 0.8, 0.2], src="@meshLoader")

    return mo


DEFAULT_CONFIG = {"scene": "CTR",
                  "deterministic": True,
                  "source": [-150, 0, 30],
                  "target": [0, 0, 30],
                  "mesh": "mesh/nasal_cavity.obj",
                  "scale": 30,
                  "rotation": [140.0, 0.0, 0.0],
                  "translation": [0.0, 0.0, 0.0],
                  "goalList": [[0.0, 0.0, 10.0]],
                  "scale_factor": 10,
                  "timer_limit": 50,
                  "timeout": 30,
                  "display_size": (1600, 800),
                  "render": 1,
                  "save_data": False,
                  "save_path": path + "/Results" + "/CTR",
                  "planning": True,
                  "discrete": True,
                  "seed": 0,
                  "start_from_history": None,
                  "python_version": "python3.7",
                  "zFar": 5000,
                  }


def createScene(root, config=DEFAULT_CONFIG, tube_list=['tube_1', 'tube_2', 'tube_3'], mode='simu_and_visu'):

    print("SCENE ", config)

    # Choose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    add_plugins(root)
    _, solver = add_visuals_and_solvers(root, config, simu, visu)
    root.dt = 0.01

    goal_mo = add_goal_node(root)
    for i, name in enumerate(tube_list):
        add_instrument_topology(root, name, i)

    instrument_combined_node = add_instruments_combined(root, tube_list, simu)
    color = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if visu:
        for i, name in enumerate(tube_list):
            add_instrument_visual(instrument_combined_node, name, color[i % len(color)], i)

    cavity_mo = add_geometry(root, config, simu, visu)

    root.addObject(RewardShaper(name="Reward", rootNode=root, goalPos=config['goalPos']))
    root.addObject(GoalSetter(name="GoalSetter", rootNode=root, goalMO=goal_mo, goalPos=config['goalPos']))

    return root
