r"""
:mod:`cape.cfdx.options.funcopts`: Options for user-defined functions
=======================================================================

This module provides the class :class:`UserFuncOpts` that parses options
for user-defined functions.

"""

# Third-party
from ...optdict import OptionsDict, BOOL_TYPES


# Main class
class UserFuncOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Options
    _optlist = (
        "args",
        "kwargs",
        "name",
        "role",
        "type",
        "verbose",
    )

    # Aliases
    _optmap = {
        "A": "args",
        "Args": "args",
        "KW": "kwargs",
        "Kwargs": "kwargs",
        "Name": "name",
        "Role": "role",
        "Type": "type",
        "Verbose": "verbose",
        "a": "args",
        "arguments": "args",
        "kw": "kwargs",
        "parameters": "args",
        "params": "args",
        "v": "verbose",
    }

    # Types
    _opttypes = {
        "kwargs": dict,
        "name": str,
        "role": str,
        "type": str,
        "verbose": BOOL_TYPES
    }

    # Allowed values
    _optvals = {
        "type": ("module", "cntl", "runner"),
    }

    # List options
    _optlistdepth = {
        "args": 1,
    }

    # Defaults
    _rc = {
        "type": "module",
        "verbose": False,
    }

    # Descriptions
    _rst_descriptions = {
        "args": "list of argument names to function",
        "kwargs": "dict of keyword arguments and names",
        "name": "name of user-defined function, including module",
        "role": "purpose of function if used with *verbose*",
        "type": "user-defined function type",
        "verbose": "option to display information during processing",
    }
