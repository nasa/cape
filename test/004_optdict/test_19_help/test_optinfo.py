
# Third-party
import numpy as np

# Local imports
from cape.optdict import OptionsDict, INT_TYPES


# Create custom class
class MyOpts(OptionsDict):
    _optcls = {
        "opt1",
        "opt2",
        "opt3",
    }

    _optmap = {
        "Opt3": "opt3",
        "option3": "opt3",
    }

    _opttypes = {
        "opt2": int,
        "opt3": INT_TYPES,
    }

    _optvals = {
        "opt3": np.arange(5),
    }

    _optlistdepth = {
        "opt2": 1,
    }

    _rc = {
        "opt3": 2,
    }

    _rst_descriptions = {
        "opt3": "option code 3",
    }


# Test help messages
def test_optinfo01():
    # Instantiate
    opts = MyOpts()
    # Display help for "opt2"
    msg2 = opts.getx_optinfo("opt2")
    vmsg2 = opts.getx_optinfo("opt2", v=True, overline=True)
    # Get first line of simple message
    line = msg2.split("\n")[0]
    # Test contents thereof
    assert line.startswith("*opt2*:")
    assert line.endswith("[:class:`int`]")
    # Test verbose message
    assert vmsg2.startswith("---")
    assert "Type:" in vmsg2
    assert ":List Depth:" in vmsg2
    # Test extra stuff didn't make it in *opt2* verbose
    assert ":Default" not in vmsg2
    # Generate even larger message
    vmsg3 = opts.getx_optinfo("opt3", v=True)
    # Test those contents slightly
    assert ":Aliases:" in vmsg3
    # Trigger the help_opt() info
    # (only can test for exceptions here)
    opts.help_opt("opt3")
    opts.help_opt("opt3", v=False)

