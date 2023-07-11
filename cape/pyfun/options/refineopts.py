r"""
:mod:`cape.pyfun.options.refineopts`: FUN3D refine options
=================================================================

Options interface for aspects of running ``Refine`` with Fun3D for
grid adaptation.

"""

# Local imports
from ...cfdx.options.util import ExecOpts
from ...optdict import BOOL_TYPES, INT_TYPES


# Class for ref cli options
class RefineOpts(ExecOpts):
    r"""Class for refine command kine settings
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
        "function",
    }

    # Types
    _opttypes = {
        "function": str,
    }

    # Allowed Values
    _optvals = {
        "function": ["translate", "distance", "loop"],
    }

    # Defaults
    _rc = {
        "function": "loop",
    }

    # Descriptions
    _rst_descriptions = {
        "function": "First argument to give to refine",
    }


# Add properties
RefineOpts.add_properties(RefineOpts._optlist, prefix="refine_")
