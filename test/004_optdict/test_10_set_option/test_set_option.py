
# Local imports
from cape import optdict
from cape.optdict import (
    WARNMODE_NONE,
    WARNMODE_QUIET,
    WARNMODE_WARN,
    WARNMODE_ERROR)
from cape.optdict.opterror import OptdictNameError


# Create a class with some option limitations
class MyOpts(optdict.OptionsDict):
    __slots__ = ()
    _optlist = {
        "opt1",
        "opt2",
        "name3",
        "name4",
    }
    _optmap = {
        "o2": "opt2"
    }
    _opttypes = {
        "name3": str,
        "name4": [list, str]
    }
    _warnmode = (WARNMODE_QUIET, WARNMODE_NONE, WARNMODE_ERROR, WARNMODE_WARN)


# Test option names
def test_optname01():
    # Simple test
    opts = MyOpts(opt1=1, o2=True, name3="a")
    assert opts["opt1"] == 1
    assert "o2" not in opts
    assert "opt2" in opts
    # Test processing of null error
    opts._process_lastwarn()


# Missing option
def test_optname02():
    # Start with empty
    opts = MyOpts()
    # No warning mode
    opts.set_opt("opt4", 1, mode=WARNMODE_NONE)
    # Assert *opt4* despite invalid name
    assert "opt4" in opts
    # Silent
    opts.set_opt("opt3", 1, mode=WARNMODE_QUIET)
    # Assert *opt3* didn't get set
    assert "opt3" not in opts
    # Warning mode
    opts.set_opt("opt3", 1, mode=WARNMODE_WARN)
    assert "opt3" not in opts
    # Error
    try:
        opts.set_opt("opt3", 1, mode=WARNMODE_ERROR)
    except OptdictNameError as err:
        # Test message
        msg = str(err)
        assert MyOpts.__name__ in msg
        assert "opt3" in msg
        assert "opt1" in msg
    else:
        # Expected an error
        assert False


# xoptlist
def test_optname03():
    # Start with empty
    opts = MyOpts()
    # Add two option names
    opts.add_xopt("opt3")
    opts.add_xopt("opt4")
    # Set allowed types
    opts.add_xopttype("opt3", int)
    opts.add_xopttype("opt4", (list, bool))
    # Test with types
    opts.set_opt("opt3", 1)


# Test option types
def test_opttype01():
    # Start with empty
    opts = MyOpts()
    # With warning
    opts.set_opt("name3", 1, mode=WARNMODE_WARN)
    # Test message
    msg = opts._lastwarnmsg
    msg1 = "'%s' option 'name3'" % MyOpts.__name__
    assert msg1 in msg
    assert "int" in msg
    assert "str" in msg
    # Test mode 1
    opts.set_opt("name3", 1, mode=WARNMODE_QUIET)
    assert "name3" not in opts
    # No warning
    opts.set_opt("name3", 1, mode=WARNMODE_NONE)
    # Here *name3* did get set b/c types were not checked
    assert "name3" in opts


# Test setel()
def test_setoptj01():
    # Test values
    opt1a = 1
    opt1b = 2
    # Start with basic options
    opts = MyOpts(opt1=opt1a)
    # Set option for a phase
    opts.set_opt("opt1", opt1b, j=2)
    # Test output
    assert opts["opt1"] == [opt1a, opt1a, opt1b]


# Test recursive type checking
def test_opttype02():
    # Empty
    opts = MyOpts()
    # Valid phased inputs
    opts.set_opt("name3", ["first", "last"])
    # Error on phase 1
    opts.set_opt("name3", ["first", 2], mode=WARNMODE_WARN)
    # Test message
    msg = opts._lastwarnmsg
    assert "int" in msg
    assert "phase 1" in msg
    # Same w/ mode=1
    opts.set_opt("name3", ["first", 2], mode=1)
    assert "name3" != ["first", 2]


# Test warning mode access
def test_warnmode01():
    # Empty
    opts = MyOpts()
    # Get warning mode from class
    i = optdict.INDEX_ONAME
    assert opts._get_warnmode(None, i) == MyOpts._warnmode[i]
    # Get global default as fallback
    MyOpts._warnmode = None
    assert opts._get_warnmode(None, 0) == optdict.DEFAULT_WARNMODE
