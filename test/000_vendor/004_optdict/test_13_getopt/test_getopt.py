
# Third-party
import numpy as np

# Local imports
from cape.optdict import (
    OptionsDict,
    FLOAT_TYPES,
    INT_TYPES,
    WARNMODE_NONE
)


# Subsection class to test recursion of setx_i()
class MySubOpts(OptionsDict):
    _optring = {
        "r": True,
    }
    _rc = {
        "w": 1,
        "y": 1.0,
    }


# Options class
class MyOpts(OptionsDict):
    _opttypes = {
        "s": MySubOpts,
        "v": FLOAT_TYPES,
    }
    _rc = {
        "v": 1.0,
    }
    _optlistdepth = {
        "a": 1,
    }


# Options
MYOPTS = {
    "v": 0.0,
    "a": 2,
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
    opts.save_x(X)
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
    mach = opts.getx_xvals("mach")
    # Test
    assert mach is None
    # Save *x*
    opts.save_x(X)
    # Get all values
    mach = opts.getx_xvals("mach")
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
    opts.save_x(X)
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
    opts.save_x(X)
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
    opts.setx_parent(MYPARENT)
    # Test fallback
    assert opts.get_opt("z") == MYPARENT["z"]


# Test -> list
def test_07_getopt_list():
    # Initialize
    opts = MyOpts(a="r1")
    # Ensure list
    assert opts.get_opt("a") == ["r1"]


# Test recursion of setx_i()
def test_08_setx():
    # Initialize
    opts = MyOpts(s={})
    # Apply run matrix
    opts.save_x(X)
    # Set case index
    opts.setx_i(0)
    # Get subsection
    secopts = opts["s"]
    # Test class
    assert isinstance(secopts, MySubOpts)
    # Test passage of *i* to subsection
    assert secopts.i == 0


# Test sampling
def test_09_sampledict():
    # Initialize with *r* for "ring"
    opts = MyOpts(s={"r": ["a", "b", "c"]})
    # Sample the whole thing for *j*
    j = 4
    optsj = opts.sample_dict(opts, j=j, f=True)
    # Get value for *r*
    vj = optsj["s"]["r"]
    vr = opts["s"]["r"]
    # Test value
    assert vj == vr[j % len(vr)]


# Test sampling w/o use of OptionsDict subsection
def test_10_samplesub():
    # Initialize with section other than "s"
    opts = MyOpts(t={"u": {"v": ["a", "b", "c"]}})
    # Sample an option for *j*
    j = 4
    vj = opts.get_subopt("t", "u", j=j)["v"]
    # Get value for *r*
    vr = opts["t"]["u"]["v"]
    # Test value; "ring" option not set
    assert vj == vr[min(j, len(vr) - 1)]
    # Sample entire *t* using get_opt
    tj = opts.get_opt("t", j=j)
    # Test results
    assert tj["u"]["v"] == vj
