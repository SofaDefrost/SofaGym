# -*- coding: utf-8 -*-
import Sofa


def loaderFor(name):
    if name.endswith(".obj"):
        return "MeshObjLoader"
    elif name.endswith(".stl"):
        return "MeshSTLLoader"
    elif name.endswith(".vtk"):
        return "MeshVTKLoader"


def CollisionMesh(attachedTo=None,
                  surfaceMeshFileName=None,
                  name="collision",
                  rotation=[0.0, 0.0, 0.0],
                  translation=[0.0, 0.0, 0.0],
                  collisionGroup=None,
                  mappingType='BarycentricMapping'):
    '''

    '''

    if attachedTo is None:
        Sofa.msg_error("Cannot create a CollisionMesh that is not attached to node.")
        return None

    collisionmodel = attachedTo.addChild(name)

    if surfaceMeshFileName is None:
        Sofa.msg_error(collisionmodel, "Unable to create a CollisionMesh without a surface mesh")
        return None

    collisionmodel.addObject(loaderFor(surfaceMeshFileName), name="loader", filename=surfaceMeshFileName,
                                rotation=rotation, translation=translation)
    collisionmodel.addObject('MeshTopology', src="@loader")
    collisionmodel.addObject('MechanicalObject')
    if collisionGroup:
        collisionmodel.addObject('PointCollisionModel', group=collisionGroup)
        collisionmodel.addObject('LineCollisionModel', group=collisionGroup)
        collisionmodel.addObject('TriangleCollisionModel', group=collisionGroup)
    else:
        collisionmodel.addObject('PointCollisionModel')
        collisionmodel.addObject('LineCollisionModel')
        collisionmodel.addObject('TriangleCollisionModel')

    if mappingType is not None:
        collisionmodel.addObject(mappingType)

    return collisionmodel
