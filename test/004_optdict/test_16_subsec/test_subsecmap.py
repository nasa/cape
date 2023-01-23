
# Third-party imports


# Local imports
from cape.optdict import (
    OptionsDict,
)


# Raw options
RAW_OPTS = {
    "Definitions": {},
    "stack": {
        "Type": "PlotFM",
        "Component": "STACK_no_base",
    },
    "stack_CA": {
        "Type": "stack",
        "Coefficient": "CA",
    },
    "stack_mach": {
        "Type": "PlotFMSweep",
        "Sweep": "mach",
        "Component": "STACK_no_base",
    },
    "stack_mach_CA": {
        "Type": "stack_mach",
        "Coefficient": "CA",
    },
    "unprocessedsection": {},
}


# Generic subfigure options
class SubfigOpts(OptionsDict):
    _optlist = {
        "Type",
        "Component",
        "Coefficient",
    }


# PlotFM subfigure options
class PlotFMOpts(SubfigOpts):
    pass


# Sweep FM plot options
class PlotFMSweepOpts(SubfigOpts):
    _optlist = {
        "Sweep",
    }


# Class with value-depdendent subsections
class MyOpts(OptionsDict):
    __slots__ = ()
    _sec_cls = {
        "Definitions": OptionsDict,
    }
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "PlotFM": PlotFMOpts,
        "PlotFMSweep": PlotFMSweepOpts,
    }


# Test cls_optmap operation
def test_01_secoptmap():
    # Read options
    opts = MyOpts(RAW_OPTS)
    # Check types
    assert type(opts["stack_CA"]) is PlotFMOpts
    assert type(opts["stack_mach_CA"]) is PlotFMSweepOpts

