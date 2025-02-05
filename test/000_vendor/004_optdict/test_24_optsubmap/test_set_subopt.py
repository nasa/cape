
# Third-party
import pytest

# Local imports
from cape.optdict import OptionsDict
from cape.optdict.opterror import OptdictTypeError


# Create options for a subsection
class PlotOptions(OptionsDict):
    __slots__ = ()
    _optlist = (
        "color",
    )
    _optmap = {
        "c": "color",
    }


# Create a class with some option limitations
class MyOpts(OptionsDict):
    __slots__ = ()
    _optlist = {
        "PlotColor",
        "PlotOptions",
        "ContourOptions",
        "ContourBins",
    }
    _optsubmap = {
        "PlotColor": ("PlotOptions", "color"),
        "ContourBins": ("ContourOptions", "bins"),
    }
    _opttypes = {
        "ContourOptions": dict,
    }
    _sec_cls = {
        "PlotOptions": PlotOptions,
    }


# Test option names
def test_setsubopt01():
    # Initialize empty options
    opts = MyOpts()
    # Apply set_subopt() directly
    opts.set_subopt("PlotOptions", "c", "blue")
    opts.set_subopt("ContourOptions", "bins", 20)
    # Check results
    assert "PlotOptions" in opts
    assert "ContourOptions" in opts
    assert "color" in opts["PlotOptions"]
    assert "bins" in opts["ContourOptions"]
    assert opts["PlotOptions"]["color"] == "blue"
    assert opts["ContourOptions"]["bins"] == 20
    # Bad type
    opts.set_opt("ContourBins", 10)
    with pytest.raises(OptdictTypeError):
        opts.set_subopt("ContourBins", "bins", 10)


# Test option names
def test_optsubmap01():
    # Initialize options iwth a submap
    opts = MyOpts(PlotColor='blue', ContourBins=20)
    # Check results
    assert "PlotOptions" in opts
    assert "ContourOptions" in opts
    assert "color" in opts["PlotOptions"]
    assert "bins" in opts["ContourOptions"]
    assert opts["PlotOptions"]["color"] == "blue"
    assert opts["ContourOptions"]["bins"] == 20
