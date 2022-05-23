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


init_gmsh("Ball")

print(gmsh.model.mesh.field.setNumber.__doc__)
tag = gmsh.model.occ.addSphere(0, 0, 0, radius=1)
gmsh.option.setNumber("Mesh.MeshSizeFactor", 1)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Ball_Volumetric.vtk")

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.model.occ.synchronize()
gmsh.write("./mesh/Ball_Surface.stl")


gmsh.model.occ.synchronize()
gmsh.fltk.run()
