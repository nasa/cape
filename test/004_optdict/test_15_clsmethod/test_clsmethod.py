
# Third party
import pytest

# Local imports
from cape import optdict


# Create some classes
class MyOpts(optdict.OptionsDict):
    _optlist = {"a"}
    _rc = {
        "a": 3,
    }


class MyOpts1(MyOpts):
    _optlist = {"b", "d"}
    _rc = {
        "b": "fun"
    }
    _optvals = {
        "b": {"fun", "lame"}
    }
    _opttypes = {
        "a": (int, float),
        "e": str,
        "f": optdict.INT_TYPES,
        "g": optdict.FLOAT_TYPES
    }
    _rst_descriptions = {
        "a": "Number of apps"
    }
    _rst_types = {
        "d": ":class:`str`"
    }


# Create some getters and setters
MyOpts1.add_property("a")
MyOpts1.add_property("b", prefix="my", name="c")


# Test _rc
def test_clsmethod01():
    # Use default value
    assert MyOpts1.get_cls_key("_rc", "a") == 3
    assert MyOpts1.get_cls_key("_rc", "b") == "fun"
    # Create instance
    opts = MyOpts1()
    # Use functions
    assert opts.get_a() == 3
    assert opts.get_my_c() == "fun"
    # Use the setter
    opts.set_a(4)
    assert opts["a"] == 4


# Test _optlist
def test_clsattr01():
    assert MyOpts1.get_cls_set("_optlist") == {"a", "b", "d"}


# Test @property errors
def test_clsprop01():
    # Attempt to set getter for OptionsDict
    with pytest.raises(optdict.OptdictTypeError):
        optdict.OptionsDict.add_getter("a")
    # Attempt to set setter for OptionsDict
    with pytest.raises(optdict.OptdictTypeError):
        optdict.OptionsDict.add_setter("a")
    # Add duplicates
    with pytest.raises(optdict.OptdictAttributeError):
        MyOpts1.add_getter("a")
    with pytest.raises(optdict.OptdictAttributeError):
        MyOpts1.add_setter("a")


# Test docstrings
def test_clsprop02():
    assert MyOpts1._genr8_rst_opttypes("d") == ":class:`str`"
    assert MyOpts1._genr8_rst_opttypes("e").endswith("`str`")
    assert MyOpts1._genr8_rst_opttypes("f").endswith("`int64`")
    assert "`float`" in MyOpts1._genr8_rst_opttypes("g")
    assert MyOpts1._genr8_rst_opttypes("h") == "{``None``} | :class:`object`"
