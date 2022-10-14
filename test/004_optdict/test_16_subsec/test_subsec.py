
# Third-party imports
import pytest

# Local imports
from cape import optdict


# Create class for the subsection
class Section1AOpts(optdict.OptionsDict):
    _optlist = {
        "alfa",
        "beta"
    }
    _opttypes = {
        "beta": int
    }


# Another section
class Section2Opts(optdict.OptionsDict):
    pass


# Create c;ass fpr a sectom
class Section1Opts(optdict.OptionsDict):
    _optlist = {
        "A",
    }
    _sec_cls = {
        "A": Section1AOpts,
    }


# Create class with a subsection
class MyOpts(optdict.OptionsDict):
    _optlist = {
        "Section1"
    }
    _sec_cls = {
        "Section1": Section1Opts,
        "Section2": Section2Opts,
    }


# Add getters and setters for Section1A
Section1AOpts.add_property("alfa")
Section1AOpts.add_property("beta")

# Promote mehtods of Section1A
optdict.promote_subsec(Section1Opts, Section1AOpts, "A", skip=["set_beta"])
optdict.promote_subsec(MyOpts, Section1Opts, "Section1")
optdict.promote_subsec(MyOpts, Section1Opts, "Section1")
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
    with pytest.raises(optdict.OptdictTypeError):
        subsec.set_beta(3.1, mode=optdict.WARNMODE_ERROR)
    # Set it anyway
    subsec.set_beta(3.1, mode=optdict.WARNMODE_NONE)
    # Test failure in getter
    with pytest.raises(optdict.OptdictTypeError):
        subsec.get_beta(mode=optdict.WARNMODE_ERROR)
