# -*- coding: utf-8 -*-
"""
Animation framework focusing in ease of use.

**********
Functions:
**********

.. autosummary::

    animate
    AnimationManager
    AnimationManagerController


splib.animation.animate
***********************
.. autofunction:: animate

splib.animation.AnimationManager
********************************
.. autofunction:: AnimationManager

splib.animation.AnimationManagerController
******************************************
.. autoclass:: AnimationManagerController(Sofa.PythonScriptController)
   :members: addAnimation

********
Modules:
********

.. autosummary::
    :toctree: _autosummary

    splib.animation.easing

"""
__all__=["animate", "easing"]
from stlib3.splib.animation.animate import AnimationManager, AnimationManagerController, animate 
