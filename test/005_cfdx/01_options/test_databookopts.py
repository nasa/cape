# Local
from cape.cfdx.options import databookopts


# Some templates
OPTS1 = {
    "Components": [
        "comp1",
        "comp2",
        "comp3"
    ],
    "comp3": {
        "Type": "TriqFM",
        "ConfigFile": "config3.xml"
    },
    "comp2": {
        "Type": "PyFunc",
        "Function": "mymod.myfunc2"
    }
}


def test_dbopts1():
    # Initialize options
    opts = databookopts.DataBookOpts(OPTS1)
    # Check types
    assert isinstance(opts["comp2"], databookopts.DBPyFuncOpts)
    assert isinstance(opts["comp3"], databookopts.DBTriqFMOpts)
    return opts
