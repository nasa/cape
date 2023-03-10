# Local
from cape.cfdx.options.configopts import ConfigOpts


# Third-party
import numpy as np
import pytest


# Sample options
OPTS1 = dict(
    Components=["comp1", "comp2"],
    ConfigFile="Config.xml",
    Points={
        "MRP": [10.0, 0.0, 0.0],
    },
    RefArea=np.pi,
    RefLength={
        "comp1": 1.0,
        "_default_": 2.0,
    },
    RefSpan=0.5,
    RefPoint={
        "_default_": "MRP",
        "comp2": [2.0, 0.5, 0.0],
    })


def test_ConfigOpts1():
    # Initialize options
    opts = ConfigOpts(**OPTS1)
    # Test point
    x1 = [4.0, 0.0, 0.1]
    # Set span
    opts.set_RefSpan(0.25)
    opts.set_RefSpan(1.0, comp="comp2")
    # Test reference quantities
    assert opts.get_RefArea() == OPTS1["RefArea"]
    assert opts.get_RefLength("comp1") == 1.0
    assert opts.get_RefLength("comp2") == 2.0
    assert opts.get_RefSpan("comp1") == 0.25
    # Test other setters
    opts.set_RefArea(1.5, comp="comp2")
    opts.set_RefLength(2.5, comp="comp2")
    # Delete the spane
    opts.pop("RefSpan")
    # Test fall-back: RefSpan -> RefLength
    assert opts.get_RefSpan("comp1") == 1.0
    # Reference point
    opts.set_RefPoint(x1, "comp2")
    assert opts.get_RefPoint("comp1") == OPTS1["Points"]["MRP"]
    assert opts.get_RefPoint("comp2") == x1


def test_ConfigOpts2():
    # Initialize options
    opts = ConfigOpts(**OPTS1)
    # Direct test point
    x1 = [0.5, 1.0, 0.0]
    x2 = [2.5, 0.0, 0.0]
    # Get coordinates of a point
    assert opts.get_Point("MRP") == OPTS1["Points"]["MRP"]
    assert opts.get_Point(x1) == x1
    # Get the main point
    xMRP = opts.get_Point("MRP")
    # Set bad values
    with pytest.raises(TypeError):
        opts.set_Point(3, "MRP2")
    with pytest.raises(IndexError):
        opts.set_Point([1, 1, 0, 0], "MRP3")
    with pytest.raises(TypeError):
        opts.expand_Point(3)
    # Get bad values
    with pytest.raises(KeyError):
        opts.get_Point("MRP2")
    # Compound point expansion
    assert opts.expand_Point(x1) == x1
    assert opts.expand_Point(["MRP", x1]) == [xMRP, x1]
    assert opts.expand_Point({"a": "MRP"}) == {"a": xMRP}
    # Move the reference point
    opts.set_Point(x2, "MRP")
    # Test moved point
    assert opts.get_Point("MRP") == x2
    # Reset points
    opts.reset_Points()
    assert opts.get_Point("MRP") == xMRP
