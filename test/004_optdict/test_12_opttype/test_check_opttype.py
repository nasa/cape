
# Local imports
from cape.optdict import OptionsDict


# Customized class
class MyOpts(OptionsDict):
    _optlist = {
        "a",
        "b",
        "c",
        "d"
    }
    _opttypes = {
        "a": (list, int),
        "b": float,
        "c": (str, dict),
        "d": int,
    }


# Raw options for testing (valid)
MYDICT = {
    "a": {
        "@raw": [1, 2]
    },
    "b": {
        "@expr": "$mach/2"
    },
    "c": {
        "@cons": {}
    },
    "d": {
        "@map": {
            "sky": 10,
            "_default_": 20
        },
        "key": "arch"
    }
}


def test_opttype01():
    # Above should be valid
    opts = MyOpts(MYDICT)
    # No changes should have occured
    assert opts == MYDICT


def test_opttype02():
    # Init empty
    opts = MyOpts()
    # Set an invalid @raw directive
    opts.set_opt("a", {"@raw": 1, "key": "mach"}, mode=1)
    assert "a" not in opts
    # Set an @expr directive w/ wrong type
    opts.set_opt("b", {"@expr": 3}, mode=1)
    assert "b" not in opts
    # Set a @map with missing "key"
    opts.set_opt("d", {"@map": {"a": 1, "b": 2}}, mode=1)
    assert "d" not in opts
    # Set a @map with "key" of wrong type
    opts.set_opt("d", {"@map": {"a": 1, "b": 2}, "key": 3}, mode=1)
    assert "d" not in opts
