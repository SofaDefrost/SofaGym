from pyquaternion import Quaternion
import numpy as np


def addSofaPlugins(node, newplugins):
    plugins = node.SofaPlugins.pluginName.value
    for plugin in newplugins.split():
        plugins.append(plugin)
    node.SofaPlugins.pluginName = plugins


def addRigidObject(node, filename, collisionFilename=None, position=[0, 0, 0, 0, 0, 0, 1], scale=[1, 1, 1],
                   textureFilename='', color=[1, 1, 1], density=0.002, name='Object', withSolver=True):

    if collisionFilename == None:
        collisionFilename = filename

    object = node.addChild(name)
    object.addObject('RequiredPlugin', name='SofaPlugins', pluginName='SofaRigid SofaLoader')
    object.addObject('MechanicalObject', template='Rigid3', position=position, showObject=False, showObjectScale=5)

    if withSolver:
        object.addObject('EulerImplicitSolver')
        object.addObject('CGLinearSolver', tolerance=1e-5, iterations=25, threshold=1e-5)
        object.addObject('UncoupledConstraintCorrection')

    visu = object.addChild('Visu')
    visu.addObject('MeshOBJLoader', name='loader', filename=filename, scale3d=scale)
    visu.addObject('OglModel', src='@loader',  color=color if textureFilename == '' else '')
    visu.addObject('RigidMapping')

    object.addObject('GenerateRigidMass', name='mass', density=density, src=visu.loader.getLinkPath())
    object.mass.init()
    translation = list(object.mass.centerToOrigin.value)
    object.addObject('UniformMass', vertexMass="@mass.rigidMass")

    visu.loader.translation = translation

    collision = object.addChild('Collision')
    collision.addObject('MeshOBJLoader', name='loader', filename=collisionFilename, scale3d=scale)
    collision.addObject('MeshTopology', src='@loader')
    collision.addObject('MechanicalObject', translation=translation)
    collision.addObject('TriangleCollisionModel')
    collision.addObject('LineCollisionModel')
    collision.addObject('PointCollisionModel')
    collision.addObject('RigidMapping')

    return object


def addEdgeCollision(parentNode, position3D, edges):
    CollisCosserat = parentNode.addChild('CollisCosserat')
    CollisCosserat.addObject('EdgeSetTopologyContainer', name="collisEdgeSet", position=position3D, edges=edges)
    CollisCosserat.addObject('EdgeSetTopologyModifier', name="collisEdgeModifier")
    CollisCosserat.addObject('MechanicalObject', name="CollisionDOFs", position=position3D)
    CollisCosserat.addObject('LineCollisionModel', bothSide="1", group='2')
    CollisCosserat.addObject('PointCollisionModel', bothSide="1", group='2')
    CollisCosserat.addObject('IdentityMapping', name="mapping")


def buildEdges(cable3DPos):
    """ This function is used to build edges required in the EdgeSetTopologyContainer component"""
    points_size = len(cable3DPos)
    edgeList = []
    for i in range(0, points_size-1):
        edgeList.append(i)
        edgeList.append(i+1)
    return edgeList


def createCosserat(parent, config, name="Cosserat", orientation=[0, 0, 0, 1], radius=0.5,
                   last_frame={"orient": [], "index": None, "dist": None}, rigidBase = None, nonColored = False,
                   buildCollision = False, youngModulus=100e6):
    [x, y, z] = config['init_pos']
    tot_length = config['tot_length']

    nbSectionS = config['nbSectionS']
    lengthS = tot_length / nbSectionS

    nbFramesF = config['nbFramesF']
    lengthF = tot_length /nbFramesF

    last_frame_orient = last_frame["orient"]
    last_frame_index = last_frame["index"]
    last_frame_dist = last_frame["dist"]

    #################################
    #            RigidBase          #
    #################################

    base = parent.addChild(name)
    base.addObject('EulerImplicitSolver', firstOrder=0, rayleighStiffness=0.2, rayleighMass=0.1)
    base.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixd", name='solver')
    base.addObject('GenericConstraintCorrection')

    if rigidBase is None:
        rigidBaseNode= base.addChild('rigidBase')

        RigidBaseMO = rigidBaseNode.addObject('MechanicalObject', template='Rigid3d',
                                                 name="RigidBaseMO", position= [x,y,z] +orientation, showObject=0,
                                                 showObjectScale=2.)
        rigidBaseNode.addObject('RestShapeSpringsForceField', name='spring', stiffness=5000,
                                   angularStiffness=5000, external_points=0, mstate="@RigidBaseMO", points=0,
                                   template="Rigid3d")
    else:
        rigidBaseNode= base.addChild(rigidBase)
        RigidBaseMO  = rigidBase.RigidBaseMO

    #################################
    #  Rate of angular Deformation  #
    #################################

    # Define: the length of each beam in a list, the positions of eahc beam
    # (flexion, torsion), the abs of each section

    positionS = []
    longeurS = []
    sum = x
    curv_abs_inputS = []
    curv_abs_inputS.append(x)

    for i in range(nbSectionS):
        positionS.append([0, 0, 0])
        longeurS.append((((i+1)*lengthS) - i*lengthS))
        sum += longeurS[i]
        curv_abs_inputS.append(sum)

    curv_abs_inputS[nbSectionS] = tot_length + x

    # Define: sofa elements
    rateAngularDeformNode = base.addChild('rateAngularDeform')
    rateAngularDeformMO = rateAngularDeformNode.addObject('MechanicalObject',
                                    template='Vec3d', name='rateAngularDeformMO',
                                    position=positionS, showIndices=0)

    BeamHookeLawForce = rateAngularDeformNode.addObject('BeamHookeLawForceField',
                                    crossSectionShape='circular', length=longeurS,
                                    radius=radius, youngModulus=youngModulus)

    #################################
    #              Frame            #
    #################################
    # Define: the abs of each frame and the position of each frame.
    framesF = []
    frames3D = []
    curv_abs_outputF = []
    for i in range(nbFramesF):
        if last_frame_index == i+1:
            sol = last_frame_dist
        else:
            sol = i * lengthF
        framesF.append([sol+x, y, z,  0, 0, 0, 1])
        frames3D.append([sol+x, y, z])
        curv_abs_outputF.append(sol+x)

    framesF.append([tot_length+x, y, z, 0, 0, 0, 1])
    frames3D.append([tot_length+x, y, z])
    curv_abs_outputF.append(tot_length+x)

    framesF = [[x, y, z, 0, 0, 0, 1]] + framesF
    frames3D = [[x, y, z]] + frames3D
    curv_abs_outputF = [x] + curv_abs_outputF

    # The node of the frame needs to inherit from rigidBaseMO and rateAngularDeform
    mappedFrameNode = rigidBaseNode.addChild('MappedFrames')
    rateAngularDeformNode.addChild(mappedFrameNode)

    framesMO = mappedFrameNode.addObject('MechanicalObject', template='Rigid3d',
                                                name="FramesMO", position=framesF,
                                            showObject=0, showObjectScale=1)

    if buildCollision:
            tab_edges = buildEdges(frames3D)
            addEdgeCollision(mappedFrameNode, frames3D, tab_edges)

    # The mapping has two inputs: RigidBaseMO and rateAngularDeformMO
    #                 one output: FramesMO
    inputMO = rateAngularDeformMO.getLinkPath()
    inputMO_rigid = RigidBaseMO.getLinkPath()
    outputMO = framesMO.getLinkPath()

    mappedFrameNode.addObject('DiscreteCosseratMapping', curv_abs_input=curv_abs_inputS,
                              curv_abs_output=curv_abs_outputF, input1=inputMO, input2=inputMO_rigid, output=outputMO,
                              debug=0, radius=1, nonColored=True)

    for i, orient in enumerate(last_frame_orient):

        lastFrame = base.addChild('lastFrame_'+str(i))
        lastFrameMo = lastFrame.addObject('MechanicalObject', template='Rigid3d', name="RigidBaseMO",
                                          position=[x+last_frame_dist, y, z] + orient, showObject=0, showObjectScale=2.)
        lastFrame.addObject("RigidRigidMapping", name="mapLastFrame", input=framesMO.getLinkPath(),
                            output=lastFrameMo.getLinkPath(), index=last_frame_index, globalToLocalCoords=True)

    return base


def express_point(base, point):
    base, point = np.array(base), np.array(point)
    pos_base, or_base = base[:3], base[3:]
    pos_point, or_point = point[:3], point[3:]
    pos = list(pos_point - pos_base)

    or_base = [or_base[3], or_base[0], or_base[1], or_base[2]]
    or_base = Quaternion(*or_base)

    or_point = [or_point[3], or_point[0], or_point[1], or_point[2]]
    or_point = Quaternion(*or_point)

    rot = list(or_base.inverse*or_point)

    coord = pos + rot[1:] + [rot[0]]
    coord = [float(p) for p in coord]

    return coord
