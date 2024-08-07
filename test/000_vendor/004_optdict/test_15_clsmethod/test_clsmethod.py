
# Third party
import pytest

# Local imports
from cape.optdict import (
    OptionsDict,
    OptdictAttributeError,
    OptdictTypeError,
    FLOAT_TYPES,
    INT_TYPES)


# Create some classes
class MyOpts(OptionsDict):
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
        "f": INT_TYPES,
        "g": FLOAT_TYPES
    }
    _rst_descriptions = {
        "a": "Number of apps"
    }
    _rst_types = {
        "d": ":class:`str`"
    }


class MyOpts2(MyOpts):
    _optlist = {"b", "c", "d", "e", "f", "g"}


# Create some getters and setters
MyOpts1.add_property("a")
MyOpts1.add_property("b", prefix="my_", name="c")
MyOpts2.add_properties(["b", "c"])
MyOpts2.add_getters(("d", "e"))
MyOpts2.add_setters(("f", "g"))


# Test _rc
def test_clsmethod01():
    # Use default value
    assert MyOpts1.getx_cls_key("_rc", "a") == 3
    assert MyOpts1.getx_cls_key("_rc", "b") == "fun"
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
    assert MyOpts1.getx_cls_set("_optlist") == {"a", "b", "d"}


# Test @property errors
def test_clsprop01():
    # Attempt to set getter for OptionsDict
    with pytest.raises(OptdictTypeError):
        OptionsDict.add_getter("a")
    # Attempt to set setter for OptionsDict
    with pytest.raises(OptdictTypeError):
        OptionsDict.add_setter("a")
    # Add duplicates
    with pytest.raises(OptdictAttributeError):
        MyOpts1.add_getter("a")
    with pytest.raises(OptdictAttributeError):
        MyOpts1.add_setter("a")


# Test docstrings
def test_clsprop02():
    assert MyOpts1._genr8_rst_opttypes("d") == ":class:`str`"
    assert MyOpts1._genr8_rst_opttypes("e").endswith("`str`")
    assert MyOpts1._genr8_rst_opttypes("f").endswith("`int`")
    assert "`float`" in MyOpts1._genr8_rst_opttypes("g")
    assert MyOpts1._genr8_rst_opttypes("h") == "{``None``} | :class:`object`"


# Test properties from list generators
def test_clsprop03():
    assert callable(MyOpts2.get_b)
    assert callable(MyOpts2.get_d)
    assert callable(MyOpts2.set_f)


# Test extra args docstring
def test_cls_getterdocstring01():
    # Extra arguments
    extra_args = {
        "category": (":class:`str`", "*a* category"),
        "subtype": "*a* subtype",
    }
    # Create a docstring for a getter w/ an extra arg for category
    docstr = MyOpts1.genr8_getter_docstring(
        "a", None, None, extra_args=extra_args)
    lines = docstr.split("\n")
    # Get first word after open-paren in signature line
    first_arg = lines[3].split("(")[1].split(",")[0]
    first_rst = lines[7].strip().split(":")[0]
    # Test signature line and first arg in :Inputs: sec
    assert first_arg == "category"
    assert first_rst == "*category*"
    # Try simple list of extra args
    docstr = MyOpts1.genr8_getter_docstring(
        "a", None, None, extra_args=["category"])
    lines = docstr.split("\n")
    # Get first word after open-paren in signature line
    first_arg = lines[3].split("(")[1].split(",")[0]
    first_rst = lines[7].strip().split(":")[0]
    # Test signature line and first arg in :Inputs: sec
    assert first_arg == "category"
    assert first_rst == "*category*"
