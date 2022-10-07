# -*- coding: utf-8 -*-
"""Create the mesh of the board.


"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "December 1 2021"

import gmsh


def init_gmsh(name="Scene"):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(name)
    gmsh.logger.start()


def _addPoint(X, Y, Z, lc=3):
    return [gmsh.model.occ.addPoint(XValue, YValue, ZValue, lc) for (XValue, YValue, ZValue) in zip(X, Y, Z)]


def _addLine(src, end, loop=True):
    LineTags = []
    NPoints = len(src)
    for i in range(NPoints):
        LineTags.append(gmsh.model.occ.addLine(src[i], end[i]))

    return LineTags


init_gmsh("Board")

BOARD_DIM = 8
HIGHEDGE = 3
SPACE = 0.15

# Step 1: Create board and cavity

BoxInTag = gmsh.model.occ.addBox(0, 0, 0, BOARD_DIM+2, BOARD_DIM+2, HIGHEDGE)
BoxOutTag = gmsh.model.occ.addBox(1, 1, 1, BOARD_DIM, BOARD_DIM, HIGHEDGE-1)

# Note: (dim, tag)
Board = gmsh.model.occ.cut([(3, BoxInTag)], [(3, BoxOutTag)])
BoardTag = Board[0][0][1]

cavity_len = (BOARD_DIM-4*SPACE)/3  # len_box - 4* interval / nb_cavity
X = [1+SPACE, 1+2*SPACE+cavity_len, 1+3*SPACE+2*cavity_len]  # edge + (n+1)*sepration + n*cavity_len
Y = [1+SPACE, 1+2*SPACE+cavity_len, 1+3*SPACE+2*cavity_len]
Z = 0.25
for i in range(3):
    for j in range(3):
        BoxTag = gmsh.model.occ.addBox(X[i], Y[j], Z, cavity_len, cavity_len, 0.65)
        CavityTag = gmsh.model.occ.cut([(3, BoardTag)], [(3, BoxTag)])

gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.5)

# Step 2: volumetric and surfacic mesh of the board
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Board_Volumetric.vtk")

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Board_Surface.stl")

# Step 3: surfacic mesh of a cavity
id = 0
for i in range(3):
    for j in range(3):
        gmsh.clear()
        BoxTag = gmsh.model.occ.addBox(X[i], Y[j], Z, cavity_len, cavity_len, 0.65)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.occ.synchronize()
        gmsh.write("./mesh/Cavity_Surface_"+str(id)+".stl")

        id += 1


gmsh.model.occ.synchronize()
gmsh.fltk.run()
