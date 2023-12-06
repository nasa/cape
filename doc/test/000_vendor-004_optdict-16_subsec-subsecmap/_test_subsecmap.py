
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
# Using _xoptkey
FULL_OPTS = dict(RAW_OPTS,
    Component="STACK_no_aft",
    Components=[
        "stack",
        "stack_CA",
        "stack_mach",
        "stack_mach_CA",
        "unprocessedsection"
    ]
)


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


# Class with some sections and some global options
class MyFullOpts(MyOpts):
    __slots__ = ()
    _optlist = {
        "Component",
        "Components",
        "Definitions",
    }
    _xoptkey = "Components"


# Test cls_optmap operation
def test_01_secoptmap():
    # Read options
    opts = MyOpts(RAW_OPTS)
    # Check types
    assert type(opts["stack_CA"]) is PlotFMOpts
    assert type(opts["stack_mach_CA"]) is PlotFMSweepOpts


# Test fuller optmap
def test_02_secoptmap():
    # Read options
    opts = MyFullOpts(FULL_OPTS)
    # Check types
    assert type(opts["stack_CA"]) is PlotFMOpts
    assert opts.get_opt("Component") == FULL_OPTS["Component"]


if __name__ == "__main__":
    test_01_secoptmap()
    test_02_secoptmap()

