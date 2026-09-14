
# Standard library
import contextlib
import os

# Third-party
import pytest
import testutils

# Local
from cape.optdict import (
    INT_TYPES,
    OptionsDict,
    OptdictKeyError,
    split_jq_path,
    truncate_jq)


# File names
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = "simple.json"


# Example of a simple subsection
class SectionOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Name
    _name = "example subsection"

    # Accepted options
    _optlist = {
        "x",
        "y",
    }

    # Types
    _opttypes = {
        "x": INT_TYPES,
    }

    # Defaults
    _rc = {
        "y": True,
    }

    # Descriptions
    _rst_descriptions = {
        "x": "x position",
        "y": "whether on",
    }


# Example of a simple options class
class MyOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Accepted options
    _optlist = {
        "a",
        "b",
        "sub",
    }

    # Aliases
    _optmap = {
        "A": "a",
    }

    # Types
    _opttypes = {
        "a": INT_TYPES,
        "b": (str, dict),
    }

    # Permitted values
    _optvals = {
        "a": (0, 1, 2),
    }

    # Defaults
    _rc = {
        "a": 1,
        "b": "fun",
    }

    # Descriptions
    _rst_descriptions = {
        "a": "first option",
    }

    # Sections
    _sec_cls = {
        "sub": SectionOpts,
    }


# Test splitting a jq path
def test_split01():
    # Test simple paths
    assert split_jq_path(".") == []
    assert split_jq_path(".a") == ["a"]
    assert split_jq_path(".a.b") == ["a", "b"]
    assert split_jq_path('.a."b c".d') == ["a", "b c", "d"]
    assert split_jq_path('.a["b c"]') == ["a", "b c"]
    # Test indices and slices
    assert split_jq_path(".a[2]") == ["a", 2]
    assert split_jq_path(".a[2:]") == ["a", slice(2, None)]
    assert split_jq_path(".a[1:3]") == ["a", slice(1, 3)]
    # Test that invalid paths raise ValueError
    with pytest.raises(ValueError):
        split_jq_path("a")
    with pytest.raises(ValueError):
        split_jq_path(".a[")
    with pytest.raises(ValueError):
        split_jq_path(".a[x]")


# Test truncation of dicts
def test_truncate01():
    # Example dict
    v = {"a": {"b": {"c": 1}}, "d": [1, 2]}
    # No truncation
    assert truncate_jq(v) == v
    # Depth 0
    assert truncate_jq(v, 0) == {}
    # Depth 1
    v1 = truncate_jq(v, 1)
    assert v1 == {"a": {}, "d": [1, 2]}


# Test item navigation
@testutils.run_sandbox(__file__, TEST_FILE)
def test_item01():
    # Bare instance
    opts0 = MyOpts()
    # Instance read from file
    opts = MyOpts(TEST_FILE)
    # Check navigation of bare sections
    assert isinstance(opts0.getx_jq_item(".sub"), SectionOpts)
    # Check navigation of values
    assert opts.getx_jq_item(".a") == 2
    assert opts.getx_jq_item(".b.c.d") == 4
    assert opts.getx_jq_item(".sub.x") == 7
    # Test alias resolution
    assert isinstance(opts.getx_jq_item(".A"), int)
    # Check missing key
    with pytest.raises(OptdictKeyError):
        opts.getx_jq_item(".bogus")
    # Scalar option w/o value has no sub-items
    with pytest.raises(OptdictKeyError):
        opts0.getx_jq_item(".b.c")


# Test help text for an option
@testutils.run_sandbox(__file__, TEST_FILE)
def test_info01_option():
    # Bare instance
    opts0 = MyOpts()
    # Instance read from file
    opts = MyOpts(TEST_FILE)
    # Get help text for option "a"
    txt = opts0.getx_jq_info(".a")
    # Test contents
    assert txt.startswith(".a\n--\n")
    assert "first option" in txt
    # No current value since no file
    assert "Current Value" not in txt
    # Check alias resolution
    txt = opts0.getx_jq_info(".A")
    assert "first option" in txt
    # File-read instance shows value
    txt = opts.getx_jq_info(".a")
    assert "Current Value" in txt
    assert "\n2\n" in txt
    # Test explicit suppression
    txt = opts.getx_jq_info(".a", showval=False)
    assert "Current Value" not in txt
    # Test missing option default
    txt = opts.getx_jq_info(".b")
    assert "Current Value" in txt
    # Test dict-valued option from file
    txt = opts.getx_jq_info(".b.c.d")
    assert "4" in txt


# Test help text for a section
@testutils.run_sandbox(__file__, TEST_FILE)
def test_info02_section():
    # Bare instance
    opts0 = MyOpts()
    # Get help text for section "sub"
    txt = opts0.getx_jq_info(".sub")
    # Test contents
    assert txt.startswith(".sub\n----\n")
    assert "example subsection" in txt
    # Recognized options listed
    assert "x: x position [int]" in txt
    assert "y: whether on" in txt
    # Test root listing
    txt = opts0.getx_jq_info(".")
    assert "Options:" in txt
    assert "Subsections:" in txt
    assert ".sub: example subsection" in txt
    assert "a: first option" in txt
    # *b* listed even though type is not set
    assert "b" in txt
    # Test aliases shown at root
    assert "A -> a" in txt
    # Test recursive expansion
    txt = opts0.getx_jq_info(".", maxdepth=1)
    assert "\n.sub\n----\n" in txt
    # Expanded subsection also lists its own options
    assert txt.count("x: x position") == 1


# Test error messages
def test_info03_errors():
    # Bare instance
    opts = MyOpts()
    # Test message for unknown option
    with pytest.raises(OptdictKeyError) as excinfo:
        opts.getx_jq_info(".sbu")
    msg = str(excinfo.value)
    assert "Close matches: sub" in msg
    assert "Available options: a, b, sub" in msg


# Test priting help text to STDOUT
def test_show_jq():
    # Bare instance
    opts = MyOpts()
    # Get help text silently
    txt_target = opts.getx_jq_info(".sub")
    # Capture STDOUT while printing
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            txt = opts.show_jq(".sub")
    # Test text
    assert txt == txt_target


if __name__ == "__main__":
    test_split01()
    test_truncate01()
    test_item01()
    test_info01_option()
    test_info02_section()
    test_info03_errors()
    test_show_jq()
