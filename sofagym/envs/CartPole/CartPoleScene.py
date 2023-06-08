import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from CartPoleToolbox import applyAction, goalSetter, rewardShaper
from sofagym.header import addVisu
from splib3.animation import AnimationManagerController
from stlib3.physics.rigid import Floor
from stlib3.scene import ContactHeader, MainHeader

#from sofagym.utils import addRigidObject

def addRigidObject(node, filename, collisionFilename=None, position=[0, 0, 0, 0, 0, 0, 1], scale=[1, 1, 1],
                   textureFilename='', color=[1, 1, 1], mass=1.0, name='Object', withSolver=True, withCollision=False):

    if collisionFilename == None:
        collisionFilename = filename

    object = node.addChild(name)
    object.addObject('RequiredPlugin', name='SofaPlugins', pluginName='SofaRigid SofaLoader')
    object.addObject('MechanicalObject', template='Rigid3', position=position, showObject=False, showObjectScale=5)

    if withSolver:
        object.addObject('EulerImplicitSolver', name='TimeIntegrationSchema')
        object.addObject('CGLinearSolver', tolerance=1e-9, iterations=200, threshold=1e-9)
        object.addObject('UncoupledConstraintCorrection')

        object.TimeIntegrationSchema.rayleighStiffness = 0.005

    visu = object.addChild('Visu')
    visu.addObject('MeshOBJLoader', name='loader', filename=filename, scale3d=scale)
    visu.addObject('OglModel', src='@loader',  color=color if textureFilename == '' else '')
    visu.addObject('RigidMapping')

    object.addObject('UniformMass', totalMass=mass)

    if withCollision:
        collision = object.addChild('Collision')
        collision.addObject('MeshOBJLoader', name='loader', filename=collisionFilename, scale3d=scale)
        collision.addObject('MeshTopology', src='@loader')
        collision.addObject('MechanicalObject')
        collision.addObject('TriangleCollisionModel')
        collision.addObject('LineCollisionModel')
        collision.addObject('PointCollisionModel')
        collision.addObject('RigidMapping')

    return object


def createScene(root, 
                config={"source": [0, 0, 160],
                        "target": [0, 0, 0],
                        "goalPos": None,
                        "seed": None,
                        "zFar":4000,
                        "init_x": 0,
                        "max_move": 24,
                        "max_angle": 0.418,
                        "dt": 0.01},
                mode='simu_and_visu'):
    
    # Choose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    # Root Parameters
    root.name = "root"
    root.dt = config['dt']

    plugins_list = ["Sofa.Component.Visual",
                    "Sofa.Component.AnimationLoop",
                    "Sofa.Component.IO.Mesh",
                    "Sofa.Component.StateContainer",
                    "Sofa.Component.Mapping.NonLinear",
                    "Sofa.Component.LinearSolver.Iterative",
                    "Sofa.Component.ODESolver.Backward",
                    "Sofa.Component.Engine.Generate",
                    "Sofa.Component.Mass",
                    "Sofa.Component.Constraint.Projective",
                    "Sofa.Component.MechanicalLoad",
                    "Sofa.Component.Constraint.Lagrangian.Correction",
                    "Sofa.Component.Constraint.Lagrangian.Model",
                    "Sofa.Component.Constraint.Lagrangian.Solver",
                    "Sofa.Component.Topology.Container.Constant",
                    "Sofa.Component.LinearSolver.Direct",
                    "Sofa.Component.Collision.Detection.Algorithm",
                    "Sofa.Component.Collision.Detection.Intersection",
                    "Sofa.Component.Collision.Response.Contact",
                    "Sofa.Component.Collision.Geometry",
                    "Sofa.GL.Component.Rendering3D",
                    "Sofa.GL.Component.Shader"]

    MainHeader(root, gravity=[0.0, -981.0, 0.0], dt=root.dt.value, plugins=plugins_list)
    ContactHeader(root, alarmDistance=0.2, contactDistance=0.09, frictionCoef=0)

    root.addObject(AnimationManagerController(root, name="AnimationManager"))
    position_spot = [[0, 0, 160]]
    direction_spot = [[0, 1, 0]]
    addVisu(root, config, position_spot, direction_spot, cutoff = 250)

    #if visu:
    #    root.VisualStyle.displayFlags = "showForceFields showVisual showBehavior showCollision"
    
    #root.addObject('CollisionPipeline', depth=6, verbose=0, draw=0)
    #root.addObject("CollisionResponse", name="CollisionResponse", response="FrictionContactConstraint")

    # Modeling
    modeling = root.addChild('Modeling')

    # Floor
    floor = Floor(modeling,
                  name="Floor",
                  translation=[0.0, -1.8, -71.6],
                  uniformScale=1,
                  isAStaticObject=True)
    
    sliding_line = floor.addChild("SlidingLine")
    sliding_line.addObject("MechanicalObject", name="points", template="Vec3", position=[[-74, -3, 0], [74, -3, 0]])
    
    x_pos = config["init_x"]
    
    # Cart
    # Parameters
    cart_position = [x_pos, 0, 0, 0, 0, 0, 1]
    cart_scale = [5, 3, 3]
    cart_color = [1, 1, 1]
    cart_mass = 1
    mesh_obj = "mesh/cube.obj"

    # Node
    cart = addRigidObject(modeling, filename= mesh_obj, collisionFilename=mesh_obj, position=cart_position, scale=cart_scale, color=cart_color, mass=cart_mass, name='Cart', withSolver=True, withCollision=True)
    
    # Constraints
    cart_constraints = cart.addChild("CartConstraints")
    cart_sliding_constraints = cart_constraints.addChild("CartSlidingConstraints")
    cart_sliding_constraints.addObject('MechanicalObject', name='sliding_point', template='Vec3', position=[0, -3, 0], showObject=False, showObjectScale=5)
    cart_sliding_constraints.addObject('RigidMapping')

    cart_pivot_constraints = cart_constraints.addChild("CartPivotConstraints")
    cart_pivot_constraints.addObject('MechanicalObject', name='pivot_point', template='Vec3', position=[0, 1, 3.15], showObject=False, showObjectScale=5)
    cart_pivot_constraints.addObject('RigidMapping')

    # Force
    cart.addObject('ConstantForceField', name="CartForce", totalForce=[0, 0, 0, 0, 0, 0])

    # Pole
    # Parameters
    pole_length = 12.5
    pole_position = [x_pos, 12, 4.3, 0, 0, 0, 1]
    pole_scale = [1, pole_length, 1]
    pole_color = [0, 1, 0]
    pole_mass = 0.1

    # Node
    pole = addRigidObject(modeling, filename= mesh_obj, collisionFilename=mesh_obj, position=pole_position, scale=pole_scale, color=pole_color, mass=pole_mass, name='Pole', withSolver=True, withCollision=True)

    # Constraints
    pole.addObject('PartialFixedConstraint', fixedDirections=[0, 0, 0, 1, 1, 0])
    pole_constraints = pole.addChild("PoleConstraints")
    pole_constraints.addObject('MechanicalObject', name='point', template='Vec3', position=[0, -11, -1.15], showObject=False, showObjectScale=5)
    pole_constraints.addObject('RigidMapping')

    # Simulation
    simulation = root.addChild('Simulation')
    simulation.addChild(modeling)
    simulation.addObject("SlidingConstraint", name="constraint1", object1=cart_sliding_constraints.sliding_point.getLinkPath(), object2=sliding_line.points.getLinkPath(), sliding_point="0", axis_1="0", axis_2="1")
    simulation.addObject("BilateralInteractionConstraint", template="Vec3", object1=cart_pivot_constraints.pivot_point.getLinkPath(), object2=pole_constraints.point.getLinkPath(), first_point="0", second_point="0")

    # SofaGym Env Components
    root.addObject(rewardShaper(name="Reward", rootNode=root, max_angle=config['max_angle'], pole_length=pole_length))
    root.addObject(goalSetter(name="GoalSetter"))
    root.addObject(applyAction(name="applyAction", root=root))
