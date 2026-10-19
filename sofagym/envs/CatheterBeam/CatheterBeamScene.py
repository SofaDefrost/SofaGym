import pathlib
import sys
from os.path import abspath, dirname

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()) + "/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from CatheterBeamToolbox import GoalSetter, RewardShaper
from sofagym.utils import addWireRestShape
from splib3.animation import AnimationManagerController

path = dirname(abspath(__file__)) + "/mesh/"


from Sofa.Units.Definitions import kg, s, mm
from Sofa.Units.UnitSystem import MechanicalUnitSystem

# Units of the scene: SOFA values (gravity, stiffness, pressure, ...) are expressed in this system.
UNITS = MechanicalUnitSystem(time=s, length=mm, mass=kg)


def add_goal_node(root):
    goal = root.addChild("Goal")
    goal.addObject("VisualStyle", displayFlags="showCollisionModels")
    goal_mo = goal.addObject(
        "MechanicalObject",
        name="GoalMO",
        showObject=True,
        drawMode="1",
        showObjectScale=2.0,
        showColor=[0, 1, 0, 0.5],
        position=[0.0, 0.0, 0.0],
    )
    return goal


def createScene(
    root,
    config={"source": [0, 0, 160], "target": [0, 0, 0], "goalPos": None, "seed": None, "zFar": 4000, "dt": 0.01},
    mode="simu_and_visu",
):

    # SETUP
    ## Choose the mode: visualization or computations (or both)
    visu, simu = False, False
    if "visu" in mode:
        visu = True
    if "simu" in mode:
        simu = True

    ## Root Parameters
    root.name = "root"
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = config["dt"]

    plugins_list = [
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.IntegrationScheme.Backward",
        "Sofa.Component.Engine.Generate",
        "Sofa.Component.Mass",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Model",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader",
        "BeamAdapter",
    ]

    root.addObject("RequiredPlugin", pluginName=plugins_list)

    root.addObject("VisualStyle", displayFlags="showVisualModels showBehaviorModels hideMappings hideForceFields")
    root.addObject("DefaultVisualManagerLoop")

    root.addObject("FreeMotionAnimationLoop")
    root.addObject("LCPConstraintSolver", mu=0.1, tolerance=1e-10, maxIt=1000, build_lcp=False)

    root.addObject("CollisionPipeline", depth=6, verbose=True, draw=False)
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject("LocalMinDistance", alarmDistance=2, contactDistance=1, angleCone=0.8, coneFactor=0.8)
    root.addObject("CollisionResponse", name="Response", response="FrictionContactConstraint")

    # SCENE
    ## Catheter
    cath = root.addChild("topoLines_cath")
    addWireRestShape(
        cath,
        name="catheterRestShape",
        length=1000.0,
        # Curved tip to steer in the vessels, as the CTR tubes (was straightLength=600 with a 4000 mm diameter spire).
        straightLength=975,
        spireDiameter=25.0,
        spireHeight=0.0,
        densityOfBeams=[40, 10],
        numEdges=200,
        numEdgesCollis=[40, 20],
        youngModulus=10000,
        youngModulusExtremity=10000,
        radius=1,
        massDensity=0.00000155,
    )
    cath.addObject("EdgeSetTopologyContainer", name="meshLinesCath")
    cath.addObject("EdgeSetTopologyModifier", name="Modifier")
    cath.addObject("EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d")
    cath.addObject("MechanicalObject", template="Rigid3d", name="dofTopo1")

    ## Guide
    guide = root.addChild("topoLines_guide")
    addWireRestShape(
        guide,
        name="GuideRestShape",
        length=1000.0,
        straightLength=980.0,
        spireDiameter=25,
        spireHeight=0.0,
        densityOfBeams=[30, 5],
        numEdges=200,
        numEdgesCollis=[50, 10],
        youngModulus=10000,
        youngModulusExtremity=10000,
        radius=0.9,
        massDensity=0.00000155,
    )
    guide.addObject("EdgeSetTopologyContainer", name="meshLinesGuide")
    guide.addObject("EdgeSetTopologyModifier", name="Modifier")
    guide.addObject("EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d")
    guide.addObject("MechanicalObject", template="Rigid3d", name="dofTopo2")

    ## Coils
    coils = root.addChild("topoLines_coils")
    addWireRestShape(
        coils,
        name="CoilRestShape",
        length=600.0,
        straightLength=540.0,
        spireDiameter=7,
        spireHeight=5.0,
        densityOfBeams=[40, 20],
        numEdges=400,
        numEdgesCollis=[30, 30],
        youngModulus=168000,
        youngModulusExtremity=168000,
        radius=0.1,
        massDensity=0.000021,
    )
    coils.addObject("EdgeSetTopologyContainer", name="meshLinesCoils")
    coils.addObject("EdgeSetTopologyModifier", name="Modifier")
    coils.addObject("EdgeSetGeometryAlgorithms", name="GeomAlgo", template="Rigid3d")
    coils.addObject("MechanicalObject", template="Rigid3d", name="dofTopo3")

    ## Combined Instrument
    instrument = root.addChild("InstrumentCombined")
    instrument.addObject("EulerImplicitIntegrationScheme", rayleighStiffness=0.2, rayleighMass=0.1, printLog=False)
    instrument.addObject("BTDLinearSolver", subpartSolve=False, verification=False, verbose=False)
    instrument.addObject(
        "RegularGridTopology",
        name="meshLinesCombined",
        # SOFA 26 update: 146 nodes, the beams of the 3 instruments (was 60, fewer than the controller needs).
        nx=146,
        ny=1,
        nz=1,
        xmin=0.0,
        xmax=1.0,
        ymin=0,
        ymax=0,
        zmin=1,
        zmax=1,
    )
    instrument.addObject("MechanicalObject", template="Rigid3d", name="DOFs", showIndices=False, ry=-90)

    instrument.addObject(
        "WireBeamInterpolation",
        name="InterpolCatheter",
        WireRestShape="@../topoLines_cath/catheterRestShape",
        printLog=False,
    )
    instrument.addObject(
        "AdaptiveBeamForceFieldAndMass",
        name="CatheterForceField",
        interpolation="@InterpolCatheter",
        massDensity=0.00000155,
    )

    instrument.addObject(
        "WireBeamInterpolation",
        name="InterpolGuide",
        WireRestShape="@../topoLines_guide/GuideRestShape",
        printLog=False,
    )
    instrument.addObject(
        "AdaptiveBeamForceFieldAndMass", name="GuideForceField", interpolation="@InterpolGuide", massDensity=0.00000155
    )

    instrument.addObject(
        "WireBeamInterpolation", name="InterpolCoils", WireRestShape="@../topoLines_coils/CoilRestShape", printLog=False
    )
    instrument.addObject(
        "AdaptiveBeamForceFieldAndMass", name="CoilsForceField", interpolation="@InterpolCoils", massDensity=0.000021
    )

    instrument.addObject(
        "InterventionalRadiologyController",
        template="Rigid3d",
        name="m_ircontroller",
        printLog=False,
        xtip=[1, 0, 0],
        step=3,
        rotationInstrument=[0, 0, 0],
        controlledInstrument=0,
        startingPos=[0, 0, 0, 0, -0.7071068, 0, 0.7071068],
        speed=0,
        instruments=["InterpolCatheter", "InterpolGuide", "InterpolCoils"],
        # Receives the keyboard events of runSofa (Ctrl + arrows, see README).
        listening=True,
    )

    instrument.addObject("LinearSolverConstraintCorrection", printLog=False, wire_optimization=True)
    instrument.addObject("FixedProjectiveConstraint", name="FixedProjectiveConstraint", indices=0)
    instrument.addObject(
        "RestShapeSpringsForceField", points="@m_ircontroller.indexFirstNode", stiffness=1e8, angularStiffness=1e8
    )

    collis = instrument.addChild("Collis", activated=True)
    collis.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
    collis.addObject("EdgeSetTopologyModifier", name="colliseEdgeModifier")
    collis.addObject("MechanicalObject", name="CollisionDOFs")
    collis.addObject(
        "MultiAdaptiveBeamMapping", name="collisMap", controller="@../m_ircontroller", useCurvAbs=True, printLog=False
    )
    # SOFA 26 update: the catheter went through the vessels as lines (tests/test_instrument_containment.py). 0.3 (wire
    # radius 0.18 + margin) is the smallest proximity keeping the instruments inside for 150 mm (0.25 does not), they
    # stay inside for 450 mm.
    collis.addObject("LineCollisionModel", proximity=0.3, group=1)
    collis.addObject("PointCollisionModel", proximity=0.3, group=1)

    cath_visu = instrument.addChild("VisuCatheter", activated=True)
    cath_visu.addObject("MechanicalObject", name="Quads")
    cath_visu.addObject("QuadSetTopologyContainer", name="ContainerCath")
    cath_visu.addObject("QuadSetTopologyModifier", name="Modifier")
    cath_visu.addObject("QuadSetGeometryAlgorithms", name="GeomAlgo", template="Vec3d")
    cath_visu.addObject(
        "Edge2QuadTopologicalMapping",
        nbPointsOnEachCircle=10,
        radius=2,
        input="@../../topoLines_cath/meshLinesCath",
        output="@ContainerCath",
        flipNormals=True,
    )
    cath_visu.addObject(
        "AdaptiveBeamMapping",
        name="VisuMapCath",
        useCurvAbs=True,
        printLog=False,
        interpolation="@../InterpolCatheter",
        input="@../DOFs",
        output="@Quads",
        isMechanical=False,
    )

    cath_visuOgl = cath_visu.addChild("VisuOgl", activated=True)
    cath_visuOgl.addObject(
        "OglModel",
        name="Visual",
        color=[0.7, 0.7, 0.7],
        quads="@../ContainerCath.quads",
        material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
    )
    cath_visuOgl.addObject("IdentityMapping", input="@../Quads", output="@Visual")

    guide_visu = instrument.addChild("VisuGuide", activated=True)
    guide_visu.addObject("MechanicalObject", name="Quads")
    guide_visu.addObject("QuadSetTopologyContainer", name="ContainerGuide")
    guide_visu.addObject("QuadSetTopologyModifier", name="Modifier")
    guide_visu.addObject("QuadSetGeometryAlgorithms", name="GeomAlgo", template="Vec3d")
    guide_visu.addObject(
        "Edge2QuadTopologicalMapping",
        nbPointsOnEachCircle=10,
        radius=0.9,  # SOFA 26 update: the guide radius, 1.8 mm (a 5.4F catheter), in a bright color (was 1, dark blue)
        input="@../../topoLines_guide/meshLinesGuide",
        output="@ContainerGuide",
        flipNormals=True,
        listening=True,
    )
    guide_visu.addObject(
        "AdaptiveBeamMapping",
        name="visuMapGuide",
        useCurvAbs=True,
        printLog=False,
        interpolation="@../InterpolGuide",
        input="@../DOFs",
        output="@Quads",
        isMechanical=False,
    )

    guide_visuOgl = guide_visu.addChild("VisuOgl")
    guide_visuOgl.addObject(
        "OglModel",
        name="Visual",
        color=[0.5, 0.7, 1.0],
        material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
        quads="@../ContainerGuide.quads",
    )
    guide_visuOgl.addObject("IdentityMapping", input="@../Quads", output="@Visual")

    coils_visu = instrument.addChild("VisuCoils", activated=True)
    coils_visu.addObject("MechanicalObject", name="Quads")
    coils_visu.addObject("QuadSetTopologyContainer", name="ContainerCoils")
    coils_visu.addObject("QuadSetTopologyModifier", name="Modifier")
    coils_visu.addObject("QuadSetGeometryAlgorithms", name="GeomAlgo", template="Vec3d")
    coils_visu.addObject(
        "Edge2QuadTopologicalMapping",
        nbPointsOnEachCircle=10,
        radius=0.45,  # SOFA 26 update: a 0.035" guidewire, in a bright color (was 0.3, green)
        input="@../../topoLines_coils/meshLinesCoils",
        output="@ContainerCoils",
        flipNormals=True,
        listening=True,
    )
    coils_visu.addObject(
        "AdaptiveBeamMapping",
        name="visuMapCoils",
        useCurvAbs=True,
        printLog=False,
        interpolation="@../InterpolCoils",
        input="@../DOFs",
        output="@Quads",
        isMechanical=False,
    )

    coils_visuOgl = coils_visu.addChild("VisuOgl")
    coils_visuOgl.addObject(
        "OglModel",
        name="Visual",
        color=[1.0, 0.8, 0.2],
        material="texture Ambient 1 0.2 0.2 0.2 0.0 Diffuse 1 1.0 1.0 1.0 1.0 Specular 1 1.0 1.0 1.0 1.0 Emissive 0 0.15 0.05 0.05 0.0 Shininess 1 20",
        quads="@../ContainerCoils.quads",
    )
    coils_visuOgl.addObject("IdentityMapping", input="@../Quads", output="@Visual")

    ## Collision
    collision = root.addChild("CollisionModel")
    collision.addObject(
        "MeshOBJLoader", name="meshLoader", filename=path + "phantom.obj", triangulate=True, flipNormals=True
    )
    collision.addObject("MeshTopology", position="@meshLoader.position", triangles="@meshLoader.triangles")
    collision.addObject("MechanicalObject", name="DOFs1", position=[0, 0, 400], scale=3, ry=90)
    collision.addObject("TriangleCollisionModel", simulated=False, moving=False)
    collision.addObject("LineCollisionModel", simulated=False, moving=False)
    collision.addObject("PointCollisionModel", simulated=False, moving=False)
    # In its own node: a node has a single state (DOFs1 here).
    collision.addChild("Visual").addObject(
        "OglModel", name="Visual", src="@../meshLoader", color=[1, 0, 0, 0.3], scale=3, ry=90
    )

    # Goal
    goal = add_goal_node(root)

    # SofaGym Env Toolbox
    root.addObject(RewardShaper(name="Reward", rootNode=root, goalPos=config["goalPos"]))
    root.addObject(GoalSetter(name="GoalSetter", rootNode=root, goal=goal, goalPos=config["goalPos"]))

    if visu:
        source = config["source"]
        target = config["target"]
        root.addObject("LightManager")
        # The spot light was at [0, zFar, 0] with a null direction (source[1] is 0): light the whole tree from the
        # camera instead (the direction of a DirectionalLight points towards the light).
        root.addObject("DirectionalLight", direction=[s - t for s, t in zip(source, target)])
        root.addObject(
            "InteractiveCamera",
            name="camera",
            position=source,
            orientation=[0.472056, -0.599521, -0.501909, 0.407217],
            # Position, orientation and lookAt over-constrain the camera: the distance to the target replaces lookAt.
            distance=sum((t - s) ** 2 for s, t in zip(source, target)) ** 0.5,
            zFar=5000,
        )

    root.addObject(AnimationManagerController(root, name="AnimationManager"))

    return root
