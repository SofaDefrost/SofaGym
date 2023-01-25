
import os

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))


from Constants import *

path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
MeshesPath = os.path.dirname(os.path.abspath(__file__))+'/mesh/'

VolumetricMeshPath = MeshesPath + 'FingerWithCavities.vtk'
SurfaceMeshPath = MeshesPath + 'FingerWithCavities.stl'


def Finger(rootNode, fixingBox, visu, simu, pullPointLocation, control1='1', control2='2', name="Finger",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0]):

    model = rootNode.addChild(name)
    if simu:
        model.addObject('EulerImplicitSolver', name='odesolver')
        model.addObject('EigenSimplicialLDLT',template='CompressedRowSparseMatrixd', name='linearSolver')

    model.addObject('MeshVTKLoader', name='loader', filename=VolumetricMeshPath, scale3d=[1, 1, 1],
                    translation=translation, rotation=rotation)
    model.addObject('TetrahedronSetTopologyContainer', position="@loader.position", tetrahedra="@loader.tetrahedra")
    model.addObject('TetrahedronSetTopologyModifier')
    model.addObject('TetrahedronSetGeometryAlgorithms', template='Vec3d')

    model.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false', showIndicesScale='4e-5')
    model.addObject('UniformMass', totalMass='0.1')
    model.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                    poissonRatio=PoissonRation,  youngModulus=YoungsModulus)

    c = model.addChild("FixedBox")
    c.addObject('BoxROI', name='BoxROI', box=fixingBox, drawBoxes=True, doUpdate=False)
    c.addObject('RestShapeSpringsForceField', points='@BoxROI.indices', stiffness='1e12')

    if simu:
        model.addObject('LinearSolverConstraintCorrection', name='GCS', solverName='@precond')

        collisionmodel = model.addChild("CollisionMesh")
        collisionmodel.addObject("MeshSTLLoader", name="loader", filename=SurfaceMeshPath,
                                 rotation=rotation, translation=translation)
        collisionmodel.addObject('MeshTopology', src="@loader")
        collisionmodel.addObject('MechanicalObject')

        collisionmodel.addObject('PointCollisionModel')
        collisionmodel.addObject('LineCollisionModel')
        collisionmodel.addObject('TriangleCollisionModel')

        collisionmodel.addObject('BarycentricMapping')

    ##########################################
    # Effector                               #
    ##########################################

    for i in range(1, 5):
        CavitySurfaceMeshPath = MeshesPath+'Cavity0' + str(i) + '.stl'
        CurrentCavity = model.addChild('Cavity0'+str(i))
        CurrentCavity.addObject('MeshSTLLoader', name='MeshLoader', filename=CavitySurfaceMeshPath, rotation=rotation,
                                translation=translation)
        CurrentCavity.addObject('MeshTopology', name='topology', src='@MeshLoader')
        CurrentCavity.addObject('MechanicalObject', src="@topology")
        CurrentCavity.addObject('BarycentricMapping', name="Mapping", mapForces="false", mapMasses="false")

    ##########################################
    # Visualization                          #
    ##########################################
    if visu:
        modelVisu = model.addChild('visu')
        modelVisu.addObject('MeshSTLLoader', filename=SurfaceMeshPath, name="loader")
        modelVisu.addObject('OglModel', src="@loader", scale3d=[1, 1, 1], translation=translation, rotation=rotation)
        modelVisu.addObject('BarycentricMapping')

    ##########################################
    # Actuation                              #
    ##########################################

    cables = model.addChild('cables')
    cable1 = cables.addChild('cable1')
    cable1.addObject('MechanicalObject', position=[[-32.5, -11, 0],
                                                   [-12.5, -11, 0],
                                                   [12.5, -11, 0],
                                                   [32.5, -11, 0],
                                                   [57.5, -11, 0],
                                                   [77, -11, 0]],
                     translation=translation,
                     rotation=rotation)
    cable1.addObject('CableConstraint', name="aCableActuator", indices="0 1 2 3 4 5", pullPoint=pullPointLocation)
    cable1.addObject('BarycentricMapping')

    # model.addObject(FingerController(node=model, name=name, control1=control1, control2=control2))
    return model
