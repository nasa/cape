
# Local imports
from cape.argread import ArgReader


# Create a class
class SubKwargs(ArgReader):
    _optlist = (
        "c",
        "d"
    )
    _opttypes = {
        "c": (float, int),
    }
    _rc = {
        "c": 1.0,
    }


class MyKwargs(SubKwargs):
    _optlist = (
        "a",
        "b",
    )
    _opttypes = {
        "a": int,
        "b": str,
    }
    _rc = {
        "a": 4,
    }
    _name = "my() parser"


# Test very simple class methods
def test_clsfn01():
    # Check names
    assert MyKwargs.get_cls_name() == MyKwargs._name


# Test main class methods for diving into bases
def test_clsfn02():
    # Get something from a subclass
    assert MyKwargs.getx_cls_key("_opttypes", "c") == SubKwargs._opttypes["c"]
    # Combine dictionaries
    data = MyKwargs.getx_cls_dict("_rc")
    assert data == dict(SubKwargs._rc, **MyKwargs._rc)

