
# Local imports
from cape import optdict
from cape.optdict.opterror import OptdictValueError


# Custom options
class MyOptions(optdict.OptionsDict):
    _optvals = {
        "a": {"on", "off"},
        "b": {1, 2, 3},
    }


# Test basic values
def test_01_valid():
    # Create options with valid values
    opts = MyOptions(a="on", b=2)
    assert opts["b"] == 2
    # Test special dicts
    opts.set_opt("b", {"@raw": {"@expr": "2"}})
    assert isinstance(opts["b"], dict)
    # Test array of valid values
    opts.set_opt("b", [1, 3, 2])
    assert opts["b"] == [1, 3, 2]


# Test unhashable option
def test_02_unhashable():
    # Create options
    opts = MyOptions(b=2)
    # Try to set inappropriate value
    opts.set_opt("b", {"n": 1})
    assert opts["b"] == 2


# Test invalid values
def test_03_invalid():
    # Create empty instance
    opts = MyOptions()
    # Try to set disallowed value
    opts.set_opt("b", 4)
    assert "b" not in opts
    # Try to set list with >0 disallowed values
    opts.set_opt("b", [1, 2, 4])


# Test close-matches thing
def test_04_close():
    # Create empty instance
    opts = optdict.OptionsDict()
    # Set some extra allowed values
    opts._xoptvals = dict(a={"blue", "brown", "green", "orange", "red"})
    # Strict checking
    opts._xwarnmode = optdict.WARNMODE_WARN
    # Try a misspelled color
    opts.set_opt("a", "greeb")
    # Get error message
    msg = opts._lastwarnmsg
    assert "green" in msg
    assert "greeb" in msg
