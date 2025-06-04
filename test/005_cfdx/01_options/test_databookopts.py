
# Third-party
import pytest

# Local
from cape.cfdx.options import databookopts


# Some templates
OPTS1 = {
    "Components": [
        "comp1",
        "comp2",
        "comp3",
        "north1",
        "south1",
    ],
    "NMin": 2000,
    "NStats": 100,
    "comp3": {
        "Type": "TriqFM",
        "ConfigFile": "config3.xml",
        "NMin": 2500,
        "NStats": 1,
        "Patches": [
            "front",
            "left",
            "top",
            "right",
            "back"
        ],
        "Targets": {
            "CN": "targ1",
        }
    },
    "comp2": {
        "Type": "PyFunc",
        "Function": "mymod.myfunc2",
        "CompID": "nozzle"
    },
    "south1": {
        "Transformations": [
            {
                "Type": "Euler321",
                "psi": "dps",
                "phi": "dph",
                "theta": "dth",
            }
        ],
    },
}

# This one with targets
OPTS2 = {
    "Components": [
        "comp1",
        "comp2"
    ],
    "comp1": {
        "Target": "targ1"
    },
    "Targets": {
        "targ1": {
            "Label": "Target #1",
        },
        "targ2": {
            "Name": "Target #2",
            "CommentChar": "#",
        },
    },
}


# Just a target
OPTS3 = {
    "Comment": "//",
    "tol": {
        "alpha": 0.05,
    }
}


def test_dbopts1():
    # Initialize options
    opts = databookopts.DataBookOpts(OPTS1)
    # Check types
    assert isinstance(opts["comp2"], databookopts.PyFuncDataBookOpts)
    assert isinstance(opts["comp3"], databookopts.TriqFMDataBookOpts)
    # Test getter function
    assert opts.get_DataBookConfigFile("comp3") == OPTS1["comp3"]["ConfigFile"]
    assert opts.get_DataBookConfigFile("comp2") is None
    # Test special *CompID*
    assert opts.get_DataBookCompID("comp1") == "comp1"
    assert opts.get_DataBookCompID("comp2") == "nozzle"
    # Test universal options and override
    assert opts.get_DataBookNMin("comp3") == 2500
    assert opts.get_DataBookNMin("comp2") == 2000
    assert opts.get_DataBookNMin("comp1") == 2000
    assert opts.get_DataBookNMin() == 2000
    # Try nonsense
    with pytest.raises(ValueError):
        opts.get_DataBookNMin(3)
    # Filter by type
    assert opts.get_DataBookType("comp1") == "FM"
    assert opts.get_DataBookByGlob("FM", "comp[0-9]") == ["comp1"]
    assert opts.get_DataBookByGlob("TriqFM") == ["comp3"]
    assert opts.get_DataBookByGlob(
        "FM", ["comp*", "north*"]) == ["comp1", "north1"]
    # Bad component type
    with pytest.raises(ValueError):
        opts.validate_DataBookType("MyFM")
    # Expected stats lists
    sts1 = ["mu"]
    sts2 = ["mu", "min", "max", "std"]
    sts3 = ["mu", "min", "max", "std", "err"]
    # Statistics for each col
    assert opts.get_DataBookColStats("comp3", "CA") == sts1
    assert opts.get_DataBookColStats("comp1", "CA") == sts3
    assert opts.get_DataBookColStats("comp1", "CL") == sts3
    assert opts.get_DataBookColStats("comp1", "x") == sts1
    assert opts.get_DataBookColStats("comp1", "mdot") == sts2
    # Test column types
    assert opts.get_DataBookFloatCols("comp1") == []
    assert opts.get_DataBookIntCols("comp1") == ["nIter", "nStats"]
    assert "nIter" in opts.get_DataBookIntCols("comp3")
    # Test full column list w/ statistical suffixes
    assert opts.get_DataBookDataCols("comp3") == opts.get_DataBookCols("comp3")
    assert "CA_std" in opts.get_DataBookDataCols("comp1")
    # Test generic "get_DataBookOpt()" fuinction
    assert opts.get_DataBookOpt("comp2", "Type") == "PyFunc"
    # Transformations
    assert len(opts.get_DataBookTransformations("south1")) == 1


def test_dbopts2_comptargets():
    # Initialize options
    opts = databookopts.DataBookOpts(OPTS1)
    # Test targets
    assert opts.get_CompTargets("comp1") == {}
    assert opts.get_CompTargets("comp3") == OPTS1["comp3"]["Targets"]


def test_dbopts3_targets():
    # Initialize options
    opts = databookopts.DataBookOpts(OPTS2)
    # Test types
    assert isinstance(
        opts["Targets"],
        databookopts.TargetDataBookCollectionOpts
    )
    assert isinstance(
        opts["Targets"]["targ1"],
        databookopts.TargetDataBookOpts
    )
    # Get targets
    targopts = opts.get_DataBookTargets()
    assert isinstance(targopts, databookopts.TargetDataBookCollectionOpts)
    # Test properties
    assert opts.get_TargetDataBookLabel("targ1") == "Target #1"
    assert opts.get_TargetDataBookLabel("targ2") == "Target #2"
    assert opts.get_TargetDataBookName("targ1") == "targ1"
    assert opts.get_TargetDataBookName("targ2") == "Target #2"
    assert opts.get_TargetDataBookCommentChar("targ2") == "#"
    # Failed properties
    with pytest.raises(ValueError):
        opts.get_TargetDataBookCommentChar("targ2a")
    # Search
    topts1 = opts["Targets"]["targ2"]
    topts2 = opts.get_TargetDataBookByName("Target #2")
    assert topts1 == topts2
    # Failed search
    with pytest.raises(KeyError):
        opts.get_TargetDataBookByName("Target 3")


def test_dbopts4_dbtargetopts():
    # Initialize options
    opts = databookopts.TargetDataBookOpts(OPTS3)
    # Test tolerance key
    assert opts.get_Tol("alpha") == OPTS3["tol"]["alpha"]
    assert opts.get_Tol("beta") == databookopts.TargetDataBookOpts._rc["tol"]
