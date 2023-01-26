
# Third party
import pytest

# Local imports
from cape.optdict import OptionsDict, OptdictAttributeError, OptdictTypeError


# Options class
class MyOpts(OptionsDict):
    _optlistdepth = {
        "b": 1,
    }


# Add some extenders
MyOpts.add_extenders(["b"])


# Test appenders
def test_extendopt01():
    # Initialize
    opts = MyOpts(b=["a", "b"], c={"a": 1})
    # Extend a list
    opts.extend_opt("b", "c")
    assert opts["b"] == ["a", "b", "c"]
    # Extend a dict
    opts.extend_opt("c", {"b": 2})
    assert opts["c"] == {"a": 1, "b": 2}
    # Extend an empty entry
    opts.extend_opt("a", 3)
    assert opts["a"] == 3
    # Null extension
    opts.extend_opt("b", None)
    assert opts["b"] == ["a", "b", "c"]
    # List extension
    opts.add_b(["d", "e"])
    assert opts["b"] == list("abcde")


# Bad extend commands
def test_extendopt02():
    # Initialize
    opts = MyOpts(b=("a", "b"), c={"a": 1})
    # Extend a tuple?
    with pytest.raises(OptdictTypeError):
        opts.add_b("c")
    # Bad extension for dict
    with pytest.raises(OptdictTypeError):
        opts.extend_opt("c", 2)
    # Add already-existing extender
    with pytest.raises(OptdictAttributeError):
        MyOpts.add_extender("b")
