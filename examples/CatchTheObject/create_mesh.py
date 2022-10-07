# -*- coding: utf-8 -*-
"""Create the mesh of the board.


"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "December 1 2021"

import gmsh
import pymeshlab


def init_gmsh(name="Scene"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(name)
    gmsh.logger.start()


def _addPoint(X, Y, Z, lc=3):
    return [gmsh.model.occ.addPoint(XValue, YValue, ZValue, lc) for (XValue, YValue, ZValue) in zip(X, Y, Z)]


def _addLine(src, end):
    LineTags = []
    NPoints = len(src)
    for i in range(NPoints):
        LineTags.append(gmsh.model.occ.addLine(src[i], end[i]))

    return LineTags


def _createQuadrilateral(X, Y, Z):
    assert len(X) == 8 and len(Y) == 8 and len(Z) == 8
    X_bottom, Y_bottom, Z_bottom  = X[:4], Y[:4], Z[:4]
    PointTags1 = _addPoint(X_bottom, Y_bottom, Z_bottom, lc = 3)
    LineTags1 = _addLine(PointTags1, [PointTags1[-1]] + PointTags1[:-1])
    WireLoop1 = gmsh.model.occ.addWire(LineTags1)
    SurfaceTag1 = gmsh.model.occ.addPlaneSurface([WireLoop1])

    X_up, Y_up, Z_up = X[4:], Y[4:], Z[4:]
    PointTags2 = _addPoint(X_up, Y_up, Z_up, lc = 3)
    LineTags2 = _addLine(PointTags2, [PointTags2[-1]] + PointTags2[:-1])
    WireLoop2 = gmsh.model.occ.addWire(LineTags2)
    SurfaceTag2 = gmsh.model.occ.addPlaneSurface([WireLoop2])

    LineTags3 = _addLine(PointTags1, PointTags2)

    SurfaceTag3 = []
    for i in range(4):
        id_botoom, id_up = LineTags1[i], LineTags2[i]
        id_left, id_right = LineTags3[(i-1) % 4],  LineTags3[i]
        WireLoop = gmsh.model.occ.addWire([id_botoom, id_left, id_up, id_right])
        SurfaceTag = gmsh.model.occ.addPlaneSurface([WireLoop])
        SurfaceTag3.append(SurfaceTag)

    SurfaceTag = SurfaceTag3[:-1] + [SurfaceTag2] + [SurfaceTag3[-1]] + [SurfaceTag1]
    SurfaceLoopTag  = gmsh.model.occ.addSurfaceLoop(SurfaceTag)
    VolumeTag = gmsh.model.occ.addVolume([SurfaceLoopTag])

    return VolumeTag


init_gmsh("Board")

X = [0, 10, 10, 0, 2, 8, 8, 2]
Y = [0, 0, 10, 10, 2, 2, 8, 8]
Z = [0, 0, 0, 0, -4, -4, -4, -4]
VolumeTag1 = _createQuadrilateral(X, Y, Z)

X = [1, 9, 9, 1, 3, 7, 7, 3]
Y = [1, 1, 9, 9, 3, 3, 7, 7]
Z = [-0.5, -0.5, -0.5, -0.5, -3, -3, -3, -3]
VolumeTag2 = _createQuadrilateral(X, Y, Z)

print(VolumeTag1, VolumeTag2)
gmsh.model.occ.cut([(3, VolumeTag1)], [(3, VolumeTag2)])

# Volumetric and surfacic mesh of the board
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Gripper_Volumetric.vtk")

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Gripper_Surface.stl")

ms = pymeshlab.MeshSet()
ms.load_new_mesh("./mesh/Gripper_Surface.stl")
ms.invert_faces_orientation()
ms.save_current_mesh("./mesh/Gripper_Surface.stl")

# Surfacic mesh of a cavity
gmsh.clear()
_createQuadrilateral(X, Y, Z)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Cavity_Surface.stl")

ms = pymeshlab.MeshSet()
ms.load_new_mesh("./mesh/Cavity_Surface.stl")
ms.invert_faces_orientation()
ms.save_current_mesh("./mesh/Cavity_Surface.stl")

# Create Cart (volumetric and surfacic)
gmsh.clear()
BoxInTag = gmsh.model.occ.addBox(0, 0, 0, 6, 4, 3)
BoxOutTag = gmsh.model.occ.addBox(0.5, 0.5, 1, 5, 3, 2)
Board = gmsh.model.occ.cut([(3, BoxInTag)], [(3, BoxOutTag)])
BoardTag = Board[0][0][1]

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Cart_Volumetric.vtk")

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Cart_Surface.stl")

ms = pymeshlab.MeshSet()
ms.load_new_mesh("./mesh/Cart_Surface.stl")
ms.invert_faces_orientation()
ms.save_current_mesh("./mesh/Cart_Surface.stl")


gmsh.model.occ.synchronize()
gmsh.fltk.run()
