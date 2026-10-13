# -*- coding: utf-8 -*-
import pytest


def pytest_collection_modifyitems(items):
    # MultiGaitRobot needs MechanicalMatrixMapperMOR, removed from ModelOrderReduction (already broken on v23.06).
    skip = pytest.mark.skip(reason="MultiGaitRobot is known to be broken (MechanicalMatrixMapperMOR was removed)")
    for item in items:
        if "multigait" in item.nodeid.lower():
            item.add_marker(skip)
