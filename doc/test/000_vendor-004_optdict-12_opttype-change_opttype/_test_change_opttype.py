
# Local imports
from cape.optdict import OptionsDict


# Class for suboptions
class MySubOpts(OptionsDict):
    _optlist = {
        "pad",
        "comp",
    }
    _opttypes = {
        "pad": float,
        "comp": str,
    }


# Customized class
class MyOpts(OptionsDict):
    _optlist = {
        "a",
    }
    _opttypes = {
        "a": MySubOpts,
    }


# Raw options for testing (valid)
MYDICT = {
    "a": [
        {
            "pad": 1.5,
            "comp": "fin",
        },
    ]
}


def test_optdicttype01():
    # Above should be valid
    opts = MyOpts(MYDICT)
    # Now *a* should be set w/o warning, but as MySubOpts, not dict
    assert "a" in opts
    assert isinstance(opts["a"], list)
    assert isinstance(opts["a"][0], MySubOpts)
    # Use OptionsDict interface to opts["a"]
    assert opts["a"][0].get_opt("comp") == "fin"
