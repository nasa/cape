
# Third-party
import pytest

# Local imports
from argread import ArgReader
from argread.errors import (
    ArgReadKeyError,
    ArgReadNameError,
    ArgReadTypeError,
    ArgReadValueError,
    assert_isinstance)


# Create a subclass
class F1Kwargs(ArgReader):
    _optlist = (
        "a",
        "b",
        "c",
        "f",
    )
    _arglist = (
        "d",
        "a",
    )
    _optlistreq = (
        "a",
    )
    _optmap = {
        "A": "a",
    }
    _rawopttypes = {
        "a": (int, str),
    }
    _opttypes = {
        "b": str,
        "c": int,
    }
    _optvals = {
        "b": ("north", "east", "south", "west"),
    }
    _optvalmap = {
        "f": {
            0: "small",
            1: "big",
        },
    }
    _optconverters = {
        "a": int,
    }


# More options, to get close matches
class F2Kwargs(ArgReader):
    _optlist = (
        "name1",
        "name2",
        "address",
    )
    _optconverters = {
        "name2": 1,
    }


# Positional parameters only
class F3Kwargs(ArgReader):
    _arglist = (
        "a",
        "b",
    )
    _opttypes = {
        "a": int,
        "b": str,
    }
    _nargmin = 2
    _nargmax = 3


# Positional parameters with checks
class F4Kwargs(ArgReader):
    _arglist = (
        "a",
        "b",
    )
    _rawopttypes = {
        "_default_": (int, str),
    }
    _optconverters = {
        "_default_": int,
    }
    _opttypes = {
        "_arg_default_": int,
    }
    _optvals = {
        "a": (0, 1),
    }


# Test some basic calls
def test_f1():
    # Instantiate valid options
    opts = F1Kwargs(A="1", b="north", c=10, f=0)
    # Test results
    assert "a" in opts
    assert isinstance(opts["a"], int)
    assert opts["a"] == 1
    # Check the _optvalmap
    assert opts["f"] == "small"
    # Parse some positional parameters
    opts = F1Kwargs(2, 3)
    # Second parameter gets mapped to kwarg
    assert "a" in opts
    assert opts["a"] == 3
    # First parameter saved as such
    assert opts.argvals[0] == 2
    assert opts.get_argvals() == (2, 3)
    # Null check of types
    assert_isinstance(3, None)


# Test some failures
def test_f1_errors():
    # Invalid option
    with pytest.raises(ArgReadNameError):
        F2Kwargs(name3="kwparse")
    # Invalid raw type
    with pytest.raises(ArgReadTypeError):
        F1Kwargs(a=1.0)
    # Invalid type
    with pytest.raises(ArgReadTypeError):
        F1Kwargs(c=1.1)
    # Invalid value
    with pytest.raises(ArgReadValueError):
        F1Kwargs(b="northwest")
    # Invalid converter
    with pytest.raises(ArgReadTypeError):
        F2Kwargs(name2="nw")
    # Missing required parameter
    with pytest.raises(ArgReadKeyError):
        # Create args
        opts = F1Kwargs(c=2)
        # Attempt to get kwarg dict, but missing *a*
        opts.get_kwargs()


# Test positional parameters
def test_f3_args():
    # Not enough args
    with pytest.raises(ArgReadTypeError):
        F3Kwargs(1)
    # Too many args
    with pytest.raises(ArgReadTypeError):
        F3Kwargs(1, "kwparse", 2, 3)
    # Unlabeled arg
    opts = F3Kwargs(1, "kwparse", 2)
    # Get args
    args = opts.get_argvals()
    # Test result
    assert len(args) == 3
    assert args[2] == 2


# Test more positoinal param versions
def test_f4_args():
    # Parse generic list
    opts = F4Kwargs(0, 1, '2')
    # Get args
    args = opts.get_argvals()
    # Assert values
    assert args[2] == 2
    # Invalid value for first parameter, 'a'
    with pytest.raises(ArgReadValueError):
        F4Kwargs(2)
