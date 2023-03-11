
# Third-party
import pytest

# Local
from cape.cfdx.options import databookopts


# Some templates
OPTS1 = {
    "Components": [
        "comp1",
        "comp2",
        "comp3"
    ],
    "NMin": 2000,
    "comp3": {
        "Type": "TriqFM",
        "ConfigFile": "config3.xml",
        "NMin": 2500,
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
    }
}


def test_dbopts1():
    # Initialize options
    opts = databookopts.DataBookOpts(OPTS1)
    # Check types
    assert isinstance(opts["comp2"], databookopts.DBPyFuncOpts)
    assert isinstance(opts["comp3"], databookopts.DBTriqFMOpts)
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


def test_dbopts2_comptargets():
    # Initialize options
    opts = databookopts.DataBookOpts(OPTS1)
    # Test targets
    assert opts.get_CompTargets("comp1") == {}
    assert opts.get_CompTargets("comp3") == OPTS1["comp3"]["Targets"]
