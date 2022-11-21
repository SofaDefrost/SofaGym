# -*- coding: utf-8 -*-
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))



from stlib3.scene import Scene as stScene
from splib3.objectmodel import setData


def Modelling(parent):
    """Create an empty node for modelling"""
    modeling = parent.addChild("Modelling")
    return modeling


def Simulation(parent):
    """Create an empty node for simulation"""
    simulation = parent.addChild("Simulation")
    simulation.addObject("EulerImplicitSolver")
    simulation.addObject("CGLinearSolver", iterations=250, tolerance=1e-20, threshold=1e-20)
    return simulation


def Scene(parent, **kwargs):
    import os
    if "plugins" not in kwargs:
        kwargs["plugins"] = []

    kwargs["plugins"].append("SofaSparseSolver")

    scene = stScene(parent, **kwargs)
    setData(scene, dt=0.025)
    setData(scene, gravity=[0., -9810., 0.])
    setData(scene.VisualStyle, displayFlags="showBehavior showForceFields")

    Modelling(scene)
    Simulation(scene)
    parent.addObject("FreeMotionAnimationLoop")
    parent.addObject("GenericConstraintSolver", maxIterations=250, tolerance=1e-20)

    # ctx = scene.Config
    # ctx.addObject("MeshSTLLoader", name="loader", filename=getLoadingLocation("data/mesh/blueprint.stl", __file__))
    # ctx.addObject("OglModel", src="@loader")
    # ctx.addObject("AddDataRepository", path=os.path.abspath(os.path.dirname(__file__)))

    return parent
