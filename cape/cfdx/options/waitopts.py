r"""
:mod:`cape.cfdx.options.waitopts`: Options for ``cape --wait``
==============================================================

This module contains options for controlling the CAPE polling function,
``cape-wait``, which waits for a certain number of cases to require
the user or agent to take action. It governs the ``"Wait"`` section of
the main CAPE JSON file.

Key options include:

    * ``"Interval"``: time to wait between polls
    * ``"NCase"``: number of cases requiring action
"""

# Local imports
from ...optdict import OptionsDict, FLOAT_TYPES, INT_TYPES


# Options for the args
class WaitArgOpts(OptionsDict):
    r"""Permissible options for the *Args* section of *Wait*"""

    # No attributes
    __slots__ = ()

    # Options
    _optlist = (
        "cons",
        "me",
        "u",
        "unmarked",
    )

    # Types
    _opttypes = {
        "cons": str,
        "me": bool,
        "u": str,
        "unmarked": bool,
    }

    # Defaults
    _rc = {
        "me": True,
        "unmarked": True,
    }

    # Descriptions
    _rst_descriptions = {
        "cons": "Additional constraints for cases to poll",
        "me": "Limit cases to those owned by this user",
        "u": "Pretend to be another user",
        "unmarked": "Limit to cases that have not been approved or failed",
    }


# Options class
class WaitOpts(OptionsDict):
    r"""Options for CAPE's "wait", which polls for *n* actionable cases

    Usually defined from the *Wait* section of a CAPE JSON file.

    :Call:
        >>> opts = WaitOpts(fname, **kw)
        >>> opts = WaitOpts(a, **kw)
        >>> opts = WaitOpts(**kw)
    :Inputs:
        *fname*: :class:`str`
            Name of JSON (or YAML) file to read from
        *a*: :class:`dict`
            Existing options to parse and check
    """
    # No attributes
    __slots__ = ()

    # Options
    _optlist = (
        "Args",
        "Interval",
        "NCase",
        "StatusList",
        "Timeout",
    )

    # Aliases
    _optmap = {
        "N": "NCase",
        "Status": "StatusList",
        "Time": "Interval",
        "args": "Args",
        "n": "NCase",
        "status": "StatusList",
        "timeout": "Timeout",
    }

    # Types
    _opttypes = {
        "Interval": FLOAT_TYPES + INT_TYPES,
        "NCase": INT_TYPES,
        "StatusList": str,
        "Timeout": FLOAT_TYPES + INT_TYPES,
    }

    # List-like options
    _optlistdepth = {
        "StatusList": 1,
    }

    # Defaults
    _rc = {
        "Interval": 60,
        "NCase": 5,
        "Timeout": 86400,
    }

    # Subsections
    _sec_cls = {
        "Args": WaitArgOpts,
    }


# Add helpers
WaitOpts.add_properties(WaitOpts._optlist, prefix="Wait")
