
# Local imports
from cape.optdict import OptionsDict, WARNMODE_ERROR


# Make a class with an options key
class MyOpts(OptionsDict):
    _optmap = {
        "comps": "Components",
    }
    _optlist = {
        "Components"
    }
    _xoptkey = "Components"


def test_01_xoptkey():
    # Instantiate
    opts = MyOpts(comps=["c1", "c2", "c3"])
    # Make sure it's allowed
    opts.set_opt("c1", 3, mode=WARNMODE_ERROR)
    # Test _xoptlist
    assert "c2" in opts._xoptlist
    # Add more options to allow
    opts.add_xopts(("c3", "c4"))
    # Test _xoptlist
    assert "c4" in opts._xoptlist

