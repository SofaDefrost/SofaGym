# -*- coding: utf-8 -*-
"""
Templates to ease collision and contact handling.

**Content:**

.. autosummary::

   CollisionMesh

|

stlib.physics.collision.CollisionMesh
*************************************
.. autofunction:: CollisionMesh

"""

from stlib3.stlib.physics.collision.collision import CollisionMesh


def FrictionalContact(applyTo=None):
    applyTo.addObject('CollisionResponse', response="FrictionContact", responseParams="mu=0")
    applyTo.addObject('LocalMinDistance', name="Proximity", alarmDistance="3", contactDistance="1")
