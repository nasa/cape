
# Third-party imports
import pytest

# Local imports
from cape.optdict import (
    OptionsDict,
    promote_subsec,
    OptdictTypeError,
    WARNMODE_ERROR,
    WARNMODE_NONE
)


# Create class for the subsection
class Section1AOpts(OptionsDict):
    _optlist = {
        "alfa",
        "beta"
    }
    _opttypes = {
        "beta": int
    }


# Create class for a section
class Section1Opts(OptionsDict):
    _optlist = {
        "A",
    }
    _sec_cls = {
        "A": Section1AOpts,
    }


# Another section
class Section2Opts(OptionsDict):
    pass


class Section3Opts(OptionsDict):
    pass


# Create class with a subsection
class MyOpts(OptionsDict):
    _optlist = {
        "Section1"
    }
    _sec_cls = {
        "Section1": Section1Opts,
        "Section2": Section2Opts,
    }


class MyOpts2(MyOpts):
    _sec_cls = {
        "Section3": Section3Opts,
    }


# Add getters and setters for Section1A
Section1AOpts.add_property("alfa")
Section1AOpts.add_property("beta")

# Promote mehtods of Section1A
promote_subsec(Section1Opts, Section1AOpts, "A", skip=["set_beta"])
MyOpts.promote_sections(skip="Section2")
MyOpts.promote_subsec(Section1Opts, "Section1")
MyOpts.promote_subsec(Section2Opts)


# Test subsection
def test_subsec01():
    # Create instance of Section1A
    opts = Section1AOpts()
    # Apply setters
    opts.set_alfa(2)
    opts.set_beta(3)
    # Test values
    assert opts["alfa"] == 2
    assert opts["beta"] == 3


# Test direct access to subsection
def test_subsec02():
    # Create empty instance
    opts = MyOpts()
    # Set option directly
    opts.set_alfa(2)
    # Assert that it worked
    assert opts["Section1"]["A"]["alfa"] == 2
    # Test opts.name feature in subsection
    subsec = opts["Section1"]["A"]
    subsec.set_opt("alpha", 2)
    assert "Section1 > A" in subsec._lastwarnmsg
    # Test failure in setter
    with pytest.raises(OptdictTypeError):
        subsec.set_beta(3.1, mode=WARNMODE_ERROR)
    # Set it anyway
    subsec.set_beta(3.1, mode=WARNMODE_NONE)
    # Test failure in getter
    with pytest.raises(OptdictTypeError):
        subsec.get_beta(mode=WARNMODE_ERROR)


# Test compound _sec_cls
def test_subsec03():
    # Create empty instance
    opts = MyOpts2()
    # Assert all sections present
    assert "Section1" in opts
    assert "Section3" in opts


# Test recursive set_x()
def test_subsec04x():
    # Create empty instance
    opts = MyOpts()
    # Some conditions
    x = {"mach": 0.5}
    # Apply (recursively?)
    opts.set_x(x)
    # Test
    assert opts["Section1"].x is x

