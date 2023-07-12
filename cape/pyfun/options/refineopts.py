r"""
:mod:`cape.pyfun.options.refineopts`: FUN3D refine options
=================================================================

Options interface for aspects of running ``Refine`` with Fun3D for
grid adaptation.

"""

# Local imports
from ...cfdx.options.util import ExecOpts
from ...optdict import BOOL_TYPES, INT_TYPES, FLOAT_TYPES


# Class for ref cli options
class RefineOpts(ExecOpts):
    r"""Class for refine command line settings
     :Inputs:
        *kw*: :class:`dict`
            Dictionary of refine command-line options
    :Outputs:
        *opts*: :class:`RefineOpts`
            refine options interface
    :Versions:
        * 2023-06-29 ``@jmeeroff``: Version 1.0
    """
    __slots__ = ()

    # Accepted options
    _optlist = {
        "initial_complexity",
        "ramp_complexity",
        "target_complexity",
    }

    # Types
    _opttypes = {
        "initial_complexity": INT_TYPES,
        "ramp_complexity": INT_TYPES,
        "target_complexity": INT_TYPES,
    }

    # Allowed Values
    _optvals = {
    }

    # Defaults
    _rc = {
        "run": False,
    }

    # Descriptions
    _rst_descriptions = {
        "initial_complexity": "First adaptation target complexity",
        "ramp_complexity": "Amount to increase target complexity",
        "target_complexity": "Final adaptaion target complexity",
    }


# Add properties
RefineOpts.add_properties(RefineOpts._optlist, prefix="refine_")
