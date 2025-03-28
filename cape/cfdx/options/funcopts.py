r"""
:mod:`cape.cfdx.options.funcopts`: Options for user-defined functions
=======================================================================

This module provides the class :class:`UserFuncOpts` that parses options
for user-defined functions.

"""

# Third-party
from ...optdict import OptionsDict


# Main class
class UserFuncOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Options
    _optlist = (
        "Args",
        "Name",
        "Type",
    )

    # Types
    _opttypes = {
        "Args": str,
        "Name": str,
        "Type": str,
    }

    # Descriptions
    _rst_descriptions = {
        "Args": "list of argument names to function",
        "Name": "name of user-defined function, including module",
        "Type": "user-defined function type",
    }
