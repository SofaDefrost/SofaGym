#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:20:50 2019

@author: stefan
"""

FingerMesh = 'FingerWithCavities_NewIteration_Full.step'
# FingerMesh = 'FingerWithCavitiesV6.step'
Volumetric_CharacteristicLengthFactor = 1
Volumetric_CharacteristicLengthMax = 5
Volumetric_CharacteristicLengthMin = 0.1

Surface_CharacteristicLengthFactor = 0.5
Surface_CharacteristicLengthMax = 3
Surface_CharacteristicLengthMin = 0.1

# Partial Rigidification

TipBoxCoords = [-71, -71, 49.5, 71, 71, 52.5]
MainBodyCoords = [-71, -71, -10, 71, 71, 49.5]

# FixedBod
FixedBoxCoords = [-50, -30, -30, -25, 30, 30]

# Material params

PoissonRation = 0.48
YoungsModulus = 12000+8000

# Cable locations

Radius = 30  # distance of cables from Z in XY-plane
TotalHeight = 51  # mm
RigidPartHeight = 8  # mm
SensorOffset = 1
CableAngle1 = 30
CableAngle2 = 150
CableAngle3 = 270

# Cavity Geometry

Cavity_CharacteristicLengthFactor = 0.43

# Markers

MarkerHeight = 5.5

# ModelTipHeight = 52.5 # This value is derived from the CAD-model, but also 'empirical',
# because it is difficult to know, where exactly Opti-track places the markers in height.
ModelTipHeight = 52  # This value is derived from the CAD-model, but also 'empirical',
# because it is difficult to know, where exactly Opti-track places the markers in height.
