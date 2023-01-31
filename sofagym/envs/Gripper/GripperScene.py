
import os
import SofaRuntime
import numpy as np

from splib3.animation import AnimationManagerController


import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from GripperToolbox import rewardShaper, goalSetter
from Gripper import Gripper

path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'

SofaRuntime.importPlugin("SofaComponentAll")


def add_plugins(root):
    root.addObject('RequiredPlugin', pluginName='BeamAdapter')
    root.addObject('RequiredPlugin', name='SofaOpenglVisual')
    root.addObject('RequiredPlugin', name="SofaMiscCollision")
    root.addObject('RequiredPlugin', name="SofaPython3")
    root.addObject('RequiredPlugin', name="SofaPreconditioner")
    root.addObject('RequiredPlugin', name="SoftRobots")
    root.addObject('RequiredPlugin', name="SofaConstraint")
    root.addObject('RequiredPlugin', name="SofaImplicitOdeSolver")
    root.addObject('RequiredPlugin', name="SofaLoader")
    root.addObject('RequiredPlugin', name="SofaSparseSolver")

    root.addObject('RequiredPlugin', name="SofaDeformable")
    root.addObject('RequiredPlugin', name="SofaEngine")
    root.addObject('RequiredPlugin', name="SofaMeshCollision")
    root.addObject('RequiredPlugin', name="SofaMiscFem")
    root.addObject('RequiredPlugin', name="SofaRigid")
    root.addObject('RequiredPlugin', name="SofaSimpleFem")
    return root


def add_visuals_and_solvers(root, config, visu, simu):
    if visu:

        source = config["source"]
        target = config["target"]
        root.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels '
                                                   'hideMappings hideForceFields showWireframe')
        root.addObject("LightManager")

        spotLoc = [0, 0, source[2]]
        root.addObject("SpotLight", position=spotLoc, direction=[0.0, 0.0, -np.sign(source[2])])
        root.addObject('InteractiveCamera', name='camera', position=source, lookAt=target, zFar=500)
        root.addObject('BackgroundSetting', color=[1, 1, 1, 1])
    if simu:
        root.addObject('FreeMotionAnimationLoop')
        root.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=1000)
        root.addObject('DefaultPipeline', draw=False, depth=6, verbose=False)
        root.addObject('BruteForceBroadPhase')
        root.addObject('BVHNarrowPhase')
    
        root.addObject('LocalMinDistance', contactDistance=5.0, alarmDistance=10.0, name='localmindistance',
                       angleCone=0.2)
        root.addObject('DefaultContactManager', name='Response', response='FrictionContactConstraint')

        root.addObject(AnimationManagerController(root))

    return root


def CreateObject(node, name, surfaceMeshFileName, visu, simu, translation=[0., 0., 0.], rotation=[0., 0., 0.],
                 uniformScale=1., totalMass=1., volume=1., inertiaMatrix=[1., 0., 0., 0., 1., 0., 0., 0., 1.],
                 color=[1., 1., 0.], isAStaticObject=False):

    object = node.addChild(name)

    object.addObject('MechanicalObject', name="mstate", template="Rigid3", translation2=translation,
                     rotation2=rotation, showObjectScale=uniformScale)

    object.addObject('UniformMass', name="mass", vertexMass=[totalMass, volume, inertiaMatrix[:]])

    if not isAStaticObject:
        object.addObject('UncoupledConstraintCorrection')
        object.addObject('EulerImplicitSolver', name='odesolver')
        object.addObject('CGLinearSolver', name='Solver')

    # collision
    if simu:
        objectCollis = object.addChild('collision')
        objectCollis.addObject('MeshObjLoader', name="loader", filename=surfaceMeshFileName, triangulate="true",
                               scale=uniformScale)

        objectCollis.addObject('MeshTopology', src="@loader")
        objectCollis.addObject('MechanicalObject')

        movement = not isAStaticObject
        objectCollis.addObject('TriangleCollisionModel', moving=movement, simulated=movement)
        objectCollis.addObject('LineCollisionModel', moving=movement, simulated=movement)
        objectCollis.addObject('PointCollisionModel', moving=movement, simulated=movement)

        objectCollis.addObject('RigidMapping')

    # visualization
    if visu:
        objectVisu = object.addChild("VisualModel")

        objectVisu.loader = objectVisu.addObject('MeshObjLoader', name="loader", filename=surfaceMeshFileName)

        objectVisu.addObject('OglModel', name="model", src="@loader", scale3d=[uniformScale]*3, color=color,
                             updateNormals=False)

        objectVisu.addObject('RigidMapping')

    return object


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3,
                             showColor=[0, 1, 0, 1], position=[0.0, 0.0, 0.0])
    return goal_mo


def create_scene(root,  config, visu, simu):

    add_plugins(root)
    add_visuals_and_solvers(root, config, visu, simu)

    if simu:
        root.gravity.value = [0.0, -9.81, 0.0]

    root.dt.value = 0.05

    goal_mo = add_goal_node(root)
    Gripper(root, visu, simu)
    CreateObject(root, name="Floor", surfaceMeshFileName="mesh/floor.obj", visu=visu, simu=simu, color=[1.0, 0.0, 0.0],
                 uniformScale=1.5, translation=[0.0, -160.0, 0.0], isAStaticObject=True)

    cubeNode = root.addChild("Cube")

    if simu:
        cubeNode.addObject('EulerImplicitSolver')
        cubeNode.addObject('SparseLDLSolver', name='solver')
        cubeNode.addObject('GenericConstraintCorrection', solverName='solver')

    cubeNode.addObject('MeshObjLoader', filename="mesh/cube.obj", flipNormals=False, triangulate=True,
                       name='meshLoader', scale=10.0)
    cubeNode.addObject('MeshTopology', position='@meshLoader.position', tetrahedra='@meshLoader.tetrahedra',
                       triangles='@meshLoader.triangles', drawTriangles='0')
    cubeMO = cubeNode.addObject('MechanicalObject', name="mstate", template="Vec3", scale=2.0,
                                translation=[0.0, -130.0, 0.0])
    cubeNode.addObject('UniformMass', totalMass=1.0)
    cubeNode.addObject('TriangleFEMForceField', youngModulus='4e2')

    cubeNode.addObject('TriangleCollisionModel')
    cubeNode.addObject('LineCollisionModel')
    cubeNode.addObject('PointCollisionModel')

    if visu:
        Cube_deform_Visu = cubeNode.addChild("VisualModel")

        Cube_deform_Visu.addObject('OglModel', name="model", src="@../meshLoader", color=[1., 1., 0.],
                                   updateNormals=True)

        Cube_deform_Visu.addObject('IdentityMapping')

    root.addObject(rewardShaper(name="Reward", rootNode=root, goalPos=config['goalPos'], effMO=cubeMO))
    root.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalPos=config['goalPos']))

    return root


def createScene(root, config={"source": [-600.0, -25, 100],
                              "target": [30, -25, 100],
                              "goalPos": [0, 0, 0]}, mode='simu_and_visu'):
    # Chose the mode: visualization or computations (or both)
    visu, simu = False, False
    if 'visu' in mode:
        visu = True
    if 'simu' in mode:
        simu = True

    root = create_scene(root, config, visu=visu, simu=simu)
    return root


def main():
    import SofaRuntime
    import Sofa.Gui
    SofaRuntime.importPlugin("SofaOpenglVisual")
    SofaRuntime.importPlugin("CImgPlugin")
    SofaRuntime.importPlugin("SofaBaseMechanics")
    SofaRuntime.importPlugin("SofaImplicitOdeSolver")

    root=Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)

    for iteration in range(10):
        Sofa.Simulation.animate(root, root.dt.value)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()

    print("End of simulation.")


if __name__ == '__main__':
    main()
