# -*- coding: utf-8 -*-
""" ActuatedArm for the tripod robot.
    This model is part of the SoftRobot toolkit available at:
        https://github.com/SofaDefrost/SoftRobots
    Available prefab:
        - ActuatedArm
        - ServoArm
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from stlib3.splib.numerics import vec3
from stlib3.splib.objectmodel import *
from stlib3.stlib.visuals import VisualModel
from stlib3.stlib.components import addOrientedBoxRoi

from s90servo import ServoMotor


mesh_path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'


@SofaPrefab
class ServoArm(object):
    def __init__(self, parent, mappingInput, name="ServoArm", indexInput=0,indice=0):
        """ServoArm is a reusable sofa model of a servo arm for the S90 servo motor

           Parameters:
                parent:        node where the ServoArm will be attached
                mappingInput:  the rigid mechanical object that will control the orientation of the servo arm
                indexInput: (int) index of the rigid the ServoArm should be mapped to
        """
        self.node = parent.addChild(name)
        self.node.addObject("MechanicalObject", name="dofs", size=1, template="Rigid3", showObject=False,
                            showObjectScale=5, translation2=[0, 25, 0])

        self.node.addObject('RigidRigidMapping', name="mapping", input=mappingInput.getLinkPath(), index=indexInput)

        visual = VisualModel(self.node, mesh_path + 'SG90_servoarm.stl', translation=[0., -25., 0.],
                             color=[1., 1., 1., 0.75])
        visual.model.writeZTransparent = True
        visual.addObject('RigidMapping', name="mapping")


@SofaPrefab
class ActuatedArm(object):
    """ActuatedArm is a reusable sofa model of a S90 servo motor and the tripod actuation arm.
           Parameters:
             - translation the position in space of the structure
             - eulerRotation the orientation of the structure
             - attachingTo (MechanicalObject)    a rest shape forcefield will constraint the object
                                                 to follow arm position
           Structure:
           Node : {
                name : "ActuatedArm"
                MechanicalObject     // Rigid position of the motor
                ServoMotor           // The s90 servo motor with its actuated wheel
                ServoArm             // The actuation arm connected to ServoMotor.ServoWheel
            }
    """

    def __init__(self, parent, name="ActuatedArm", translation=[0.0, 0.0, 0.0], eulerRotation=[0.0, 0.0, 0.0],
                 attachingTo=None):

        self.node = parent.addChild(name)
        self.servomotor = ServoMotor(self.node, translation=translation, rotation=eulerRotation, indice=int(name[-1]))
        self.servoarm = ServoArm(self.node, self.servomotor.ServoWheel.dofs, indice=int(name[-1]))

        # Create a public attribute and connect it to the private one.
        self.node.addData(name="angleIn", group="ArmProperties", help="angle of rotation (in radians) of the arm",
                          type="float", value=0)
        self.node.ServoMotor.getData("angleIn").setParent(self.node.getData("angleIn"))

        # Create a public attribute and connect it to the internal one.
        self.node.addData(name="angleOut", group="ArmProperties", help="angle of rotation (in radians) of the arm",
                          type="float", value=0)
        self.node.getData("angleOut").setParent(self.node.ServoMotor.getData("angleOut"))

        if attachingTo is not None:
            constraint = self.addConstraint(attachingTo, translation, eulerRotation)
            attachingTo.addObject('RestShapeSpringsForceField', name="rssff"+name,
                                  points=constraint.BoxROI.getData("indices"),
                                  external_rest_shape=constraint.dofs.getLinkPath(), stiffness='1e12')

    def addConstraint(self, deformableObject, translation, eulerRotation):
        constraint = self.node.addChild("Constraint")
        o = addOrientedBoxRoi(constraint, position=deformableObject.dofs.getData("rest_position"),
                              translation=vec3.vadd(translation, [0.0, 25.0, 0.0]),
                              eulerRotation=eulerRotation, scale=[45, 15, 30])
        o.drawSize = 1
        o.drawBoxes = False

        constraint.addObject("TransformEngine", input_position="@BoxROI.pointsInROI", translation=translation,
                             rotation=eulerRotation, inverse=True)

        constraint.addObject("MechanicalObject", name="dofs", template="Vec3d",
                             position="@TransformEngine.output_position", showObject=True, showObjectScale=5.0)

        constraint.addObject('RigidMapping', name="mapping", input=self.node.ServoMotor.ServoWheel.dofs.getLinkPath(),
                             output="@./")

        return constraint

    def addBox(self, position, translation, eulerRotation):
        constraint = self.node.addChild("Box")
        o = addOrientedBoxRoi(constraint, position=position,
                              translation=vec3.vadd(translation, [0.0, 25.0, 0.0]),
                              eulerRotation=eulerRotation, scale=[45, 15, 30])
        o.init()
