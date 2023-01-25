import Sofa
import math

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from math import cos
from math import sin
import numpy as np
from splib.animation import AnimationManagerController
from TrunkCupToolbox import rewardShaper, goalSetter


import os
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
dirPath = os.path.dirname(os.path.abspath(__file__))+'/'

def rotate(v,q):

	c0 = ((1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))*v[0] + (2.0 * (q[0] * q[1] - q[2] * q[3])) * v[1] + (2.0 * (q[2] * q[0] + q[1] * q[3])) * v[2])
	c1 = ((2.0 * (q[0] * q[1] + q[2] * q[3]))*v[0] + (1.0 - 2.0 * (q[2] * q[2] + q[0] * q[0]))*v[1] + (2.0 * (q[1] * q[2] - q[0] * q[3]))*v[2])
	c2 = ((2.0 * (q[2] * q[0] - q[1] * q[3]))*v[0] + (2.0 * (q[1] * q[2] + q[0] * q[3]))*v[1] + (1.0 - 2.0 * (q[1] * q[1] + q[0] * q[0]))*v[2])

	return [c0, c1, c2]


def normalize(x):
	norm = np.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
	for i in range(0,3):
		x[i] = x[i]/norm

def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject('VisualStyle', displayFlags="showCollisionModels")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode="1", showObjectScale=3, showColor=[0, 1, 0, 1], position=[0.0, 0.0, 0.0])
    return goal_mo

GPU = 0
INVERSE = False

def add_cable(trunk):
	length1 = 10
	length2 = 2
	lengthTrunk = 195

	pullPoint = [[0., length1, 0.], [-length1, 0., 0.], [0., -length1, 0.], [length1, 0., 0.]]
	direction = [0, length2-length1, lengthTrunk]
	normalize(direction)

	displacementL = [7.62, -18.1, 3.76, 30.29]
	displacementS = [-0.22, -7.97, 3.89, 12.03]

	nbCables = 4

	for i in range(0,nbCables):
		theta = 1.57*i
		q = [0.,0.,sin(theta/2.), cos(theta/2.)]

		position = [[0, 0, 0]]*20
		for k in range(0,20,2):
			v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21]
			position[k] = rotate(v,q)
			v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27]
			position[k+1] = rotate(v,q)

		pullPointList = [[pullPoint[i][0], pullPoint[i][1], pullPoint[i][2]]]

		cableL = trunk.addChild('cableL'+str(i))
		cableL.addObject('MechanicalObject', name='meca',position= pullPointList+ position)

		idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
		cableL.addObject('CableConstraint', template='Vec3d', name="cable", hasPullPoint= 0, indices= idx, valueType="displacement", value=displacementL[i])
		cableL.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)


		# pipes
		pipes = trunk.addChild('pipes'+str(i))
		pipes.addObject('EdgeSetTopologyContainer', position= pullPointList + position, edges= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
		pipes.addObject('MechanicalObject', name="pipesMO")
		pipes.addObject('UniformMass', totalMass=0.003)
		pipes.addObject('MeshSpringForceField', stiffness=1.5e2, damping=0, name="FF")
		pipes.addObject('BarycentricMapping', name="BM")


	for i in range(0,nbCables):
		theta = 1.57*i
		q = [0.,0.,sin(theta/2.), cos(theta/2.)]

		position = [[0, 0, 0]]*10
		for k in range(0,9,2):
			v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21]
			position[k] = rotate(v,q)
			v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27]
			position[k+1] = rotate(v,q)

		pullPointList = [[pullPoint[i][0], pullPoint[i][1], pullPoint[i][2]]]

		cableS = trunk.addChild('cableS'+str(i))
		cableS.addObject('MechanicalObject', name='meca', position=pullPointList+ position)

		idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		cableS.addObject('CableConstraint', template='Vec3d', name="cable", hasPullPoint=0, indices=idx, valueType="displacement", value=displacementS[i])
		cableS.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)


def createScene(rootNode, config={"source": [-600.0, -25, 100], "target": [30, -25, 100], "goalPos": [0, 0, 0]}, mode = 'simu_and_visu'):

	#Chose the mode: visualization or computations (or both)
	visu, simu = False, False
	if 'visu' in mode:
		visu = True
	if 'simu' in mode:
		simu = True

	if simu:
		rootNode.gravity.value = [0, -9180, 0]
	dt = 0.01
	rootNode.dt.value = dt

	rootNode.addObject('RequiredPlugin', name="SoftRobots", pluginName='SoftRobots')
	rootNode.addObject('RequiredPlugin', name="SofaPython", pluginName='SofaPython3')
	rootNode.addObject('RequiredPlugin', name='SofaOpenglVisual')
	rootNode.addObject('RequiredPlugin', name="SofaSparseSolver")
	rootNode.addObject('RequiredPlugin', name='SofaPreconditioner')
	rootNode.addObject('RequiredPlugin', name='SofaConstraint')
	rootNode.addObject('RequiredPlugin', name="SofaImplicitOdeSolver")
	rootNode.addObject('RequiredPlugin', name='SofaLoader')
	rootNode.addObject('RequiredPlugin', name='SofaBoundaryCondition')
	rootNode.addObject('RequiredPlugin', name='SofaDeformable')
	rootNode.addObject('RequiredPlugin', name="SofaEngine")
	rootNode.addObject('RequiredPlugin', name='SofaGeneralLoader')
	rootNode.addObject('RequiredPlugin', name="SofaMeshCollision")
	rootNode.addObject('RequiredPlugin', name='SofaSimpleFem')

	rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels hideForceFields showInteractionForceFields hideWireframe')

	if simu:
		rootNode.addObject('GenericConstraintSolver', maxIterations=2500, tolerance=1e-15)
		rootNode.addObject('FreeMotionAnimationLoop')
		rootNode.addObject('DefaultPipeline', verbose=0)
		rootNode.addObject('BruteForceDetection', name="N2")
		rootNode.addObject('DefaultContactManager', response="FrictionContactConstraint", responseParams="mu=0.8")
		rootNode.addObject('LocalMinDistance', name="Proximity", alarmDistance=2, contactDistance=0.5, angleCone=0, coneFactor=0.5)

	if visu:
		# rootNode.addObject('BackgroundSetting', color=[0, 0.168627, 0.211765, 1])
		rootNode.addObject('BackgroundSetting', color=[1, 1, 1, 1])
	#cylinder
	cylinder = rootNode.addChild('cylinder')
	cylinder.addObject('EulerImplicitSolver')
	cylinder.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")
	cylinder.addObject('MeshVTKLoader', name='loader', filename=path+'cup.vtk', translation=[42, -50, 40], rotation=[-90, 0, 0])

	#cylinder.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
	cylinder.addObject('TetrahedronSetTopologyContainer', position="@loader.position", tetrahedra="@loader.tetrahedra")
	cylinder.addObject('TetrahedronSetTopologyModifier')
	cylinder.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')
	cylinder.addObject('MechanicalObject', name='tetras', template='Vec3d')
	cylinder.addObject('UniformMass', totalMass=0.0024)
	cylinder.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', poissonRatio=0.3,  youngModulus=1000)
	cylinder.addObject('LinearSolverConstraintCorrection')

	#colli
	if simu:
		cylinderColli = cylinder.addChild('cylinderColli')
		cylinderColli.addObject('MeshSTLLoader', name="loader", filename=path+"cup_colli.stl", translation=[42, -50, 40], rotation=[-90, 0, 0], scale3d=[1, 1, 1])
		cylinderColli.addObject('MeshTopology', src="@loader")
		cylinderColli.addObject('MechanicalObject')
		cylinderColli.addObject('TriangleCollisionModel', group=0)
		cylinderColli.addObject('LineCollisionModel', group=0)
		cylinderColli.addObject('PointCollisionModel', group=0)
		cylinderColli.addObject('BarycentricMapping')

	#visu
	if visu:
		cylinderVisu = cylinder.addChild('cylinderVisu')
		cylinderVisu.addObject('MeshSTLLoader', filename=path+"cup.stl", translation=[42, -50, 40], rotation=[-90, 0, 0])
		cylinderVisu.addObject('OglModel', color=[0.5, 0, 0.2, 1])
		cylinderVisu.addObject('BarycentricMapping')

	#cylinder/cylinderEffector
	cylinderEffector = cylinder.addChild('cylinderEffector')
	cylinderEffector.addObject('MechanicalObject', name="effectorPoint", position=[[42, -50, 40], [42, -20, 15], [42, 8, 70],  [42, 28, 5]], drawMode=1, showColor=[255, 0, 0, 255], showObjectScale=3, showObject=1)
	cylinderEffector.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

	#trunk
	trunk = rootNode.addChild('trunk')
	trunk.addObject('EulerImplicitSolver', name='odesolver', firstOrder=0, rayleighMass=0.1, rayleighStiffness=0.1)
	trunk.addObject('EigenSimplicialLDLT', template='CompressedRowSparseMatrixd', name='linearSolver')

	trunk.addObject('MeshVTKLoader', name='loader', filename=path+'trunk2.vtk')
	trunk.addObject('TetrahedronSetTopologyContainer', position="@loader.position", tetrahedra="@loader.tetrahedra")
	trunk.addObject('TetrahedronSetTopologyModifier')
	trunk.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')

	trunk.addObject('MechanicalObject', name='tetras', rest_position="@loader.position", position="@loader.position", template='Vec3d', showIndices='false', showIndicesScale=4e-5)
	trunk.addObject('ReadState', name="state", filename=path+"TrunkGrasping_StateTriangleColliSelf", shift=0.01)
	trunk.addObject('UniformMass', totalMass=0.042)
	trunk.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', poissonRatio=0.3,  youngModulus=600)

	trunk.addObject('BoxROI', name='boxROI', box=[-10, -10, 0, 10, 10, 10], drawBoxes=True)
	trunk.addObject('PartialFixedConstraint', fixedDirections=[1, 1, 1], indices="@boxROI.indices")

	if simu:
		trunk.addObject('LinearSolverConstraintCorrection', name='GCS', solverName='precond')

	# trunk/cables
	add_cable(trunk)

	#trunk/trunkCollision
	if simu:
		for i in range(1,3):
			trunkCollision = trunk.addChild('collision'+str(i))
			trunkCollision.addObject('MeshSTLLoader', name="loader", filename=path+"trunk2_colli"+str(i)+".stl")
			trunkCollision.addObject('MeshTopology', src="@loader")
			trunkCollision.addObject('MechanicalObject')
			trunkCollision.addObject('TriangleCollisionModel', group=1)
			trunkCollision.addObject('LineCollisionModel', group=1)
			trunkCollision.addObject('PointCollisionModel', group=1)
			trunkCollision.addObject('BarycentricMapping')

	#trunk/trunkVisu
	if visu:
		trunkVisu = trunk.addChild('visu')
		trunkVisu.addObject('MeshSTLLoader', name="loader", filename=path+"trunk2.stl")
		trunkVisu.addObject('OglModel', template='Vec3d', src="@loader", color=[1., 1., 1., 1.])
		trunkVisu.addObject('BarycentricMapping')


	#boxVisu and support
	if visu:
		boxVisu = rootNode.addChild('boxVisu')
		boxVisu.addObject('MeshSTLLoader', name="loaderBox1", filename=path+"trunkBox1.stl")
		boxVisu.addObject('OglModel', name="box1", template='Vec3d', color=[0.9, 0.7, 0.5, 1.], src="@loaderBox1")
		boxVisu.addObject('MeshSTLLoader', name="loaderBox2", filename=path+"trunkBox2.stl")
		boxVisu.addObject('OglModel', name="box2", template='Vec3d', color=[1., 1., 1., 0.2], src="@loaderBox2")
		boxVisu.addObject('MeshSTLLoader', name="loaderBox3", filename=path+"trunkBox3.stl")
		boxVisu.addObject('OglModel', name="box3", template='Vec3d', color=[0.9, 0.7, 0.5, 1.], src="@loaderBox3")
		boxVisu.addObject('MeshSTLLoader', name="loaderSupport", filename=path+"trunkSupport.stl")
		boxVisu.addObject('OglModel', name="support", template='Vec3d', color=[0.2, 0.2, 0.2, 1.], src="@loaderSupport")

	boxContact = rootNode.addChild('boxContact')
	boxContact.addObject('MeshObjLoader', name='loader', filename=path+'square1.obj', scale=300, rotation=[90, 0, 180], translation=[150, -100, -20])
	boxContact.addObject('MeshTopology', src='@loader', name='topo')
	boxContact.addObject('MechanicalObject')
	boxContact.addObject('TriangleCollisionModel', group=2)
	boxContact.addObject('LineCollisionModel', group=2)
	boxContact.addObject('PointCollisionModel', group=2)

	goal_mo = add_goal_node(rootNode)
	rootNode.addObject(rewardShaper(name="Reward", rootNode=rootNode, goalPos = config['goalPos']))
	rootNode.addObject(goalSetter(name="GoalSetter", goalMO=goal_mo, goalPos = config['goalPos']))

	if simu:
		rootNode.addObject(AnimationManagerController(name="AnimationManager"))

	if visu:
		source = config["source"]
		target = config["target"]
		rootNode.addObject("LightManager")
		spotLoc = [0, 0, 1000]
		rootNode.addObject("SpotLight", position=spotLoc, direction=[0, 0.0, -1.0])

		rootNode.addObject("InteractiveCamera", name='camera', position = source, lookAt = target, zFar = 500)

	return rootNode
