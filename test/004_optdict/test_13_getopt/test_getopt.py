
# Third-party
import numpy as np

# Local imports
from cape.optdict import (
    OptionsDict,
    FLOAT_TYPES,
    INT_TYPES,
    WARNMODE_NONE
)


# Options class
class MyOpts(OptionsDict):
    _opttypes = {
        "v": FLOAT_TYPES,
    }
    _rc = {
        "v": 1.0,
    }


# Options
MYOPTS = {
    "v": 0.0,
    "w": {
        "@expr": "$mach * np.sin($aoa * $DEG)",
    },
}

# Conditions
X = {
    "mach": np.linspace(0.5, 1.5, 5),
    "aoa": np.array([6.0, 4.0, 2.0, 2.0, -1.0]),
    "DEG": np.pi / 180.0
}

# Options with parent
MYPARENT = {
    "z": 1,
}


# Access condition
def test_01_getoptx():
    # Initialize
    opts = OptionsDict(MYOPTS)
    # Set conditions
    opts.set_x(X)
    # Get normal component of Mach
    w = opts.get_opt("w")
    # Calculate target
    wtarg = X["mach"] * np.sin(X["aoa"] * X["DEG"])
    # Test
    assert np.max(np.abs(w - wtarg)) <= 1e-6


# Access to run matrix
def test_02_getx():
    # Initialize
    opts = OptionsDict()
    # Test w/o opts.x
    mach = opts.get_xvals("mach")
    # Test
    assert mach is None
    # Save *x*
    opts.set_x(X)
    # Get all values
    mach = opts.get_xvals("mach")
    # Test
    assert np.max(np.abs(mach - X["mach"])) <= 1e-6


# Test phase/index compound
def test_03_getoptj():
    # Initialize
    opts = OptionsDict(
        a=[
            {"@expr": "$mach"},
            {"@expr": "10*$mach"}
        ])
    opts.set_x(X)
    # Sample phased expression
    a0 = opts.get_opt("a", i=0, j=0)
    a1 = opts.get_opt("a", i=0, j=1)
    # Test
    assert a0 == X["mach"][0]
    assert a1 == X["mach"][0] * 10


# Get an invalid value that couldn't be checked at assign
def test_04_getopt_invalid():
    # Initialize
    opts = OptionsDict(a={"@expr": "(4*$mach) // 2"})
    opts.set_x(X)
    # Set types
    opts.add_xopttype("a", INT_TYPES)
    # Try to evaluate
    assert opts.get_opt("a", i=1) is None
    # Without checks, should get 1.0
    assert opts.get_opt("a", i=1, mode=WARNMODE_NONE) == 1.0


# Test defaults
def test_05_getopt_rc():
    # Initialize
    opts = MyOpts()
    # Get a default value for *v*
    assert opts.get_opt("v") == opts._rc["v"]
    # Override with *vdef*
    assert opts.get_opt("v", vdef=2.0) == 2.0
    # Set an instance-specific default
    w = -5.
    opts._xrc = dict(w=w)
    assert opts.get_opt("w") == w
    # Override with *vdef*
    assert opts.get_opt("w", vdef=1) == 1


# Test fall-back
def test_06_parent():
    # Initialize
    opts = OptionsDict(MYOPTS)
    opts.set_parent(MYPARENT)
    # Test fallback
    assert opts.get_opt("z") == MYPARENT["z"]
