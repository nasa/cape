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
    * ``"Timeout"``: maximum total time to wait

The ``"Interval"`` and ``"Timeout"`` options may be given as a number of
seconds or as a string with a suffix, e.g. ``"60s"``, ``"30m"``,
``"2h"``, or ``"1d"``. Use :func:`convert_time` to convert either form
to a number of seconds.
"""

# Standard library
from typing import Union

# Local imports
from ...optdict import OptionsDict, FLOAT_TYPES, INT_TYPES, OptdictValueError


# Multipliers to convert time suffixes to seconds
TIME_SUFFIXES = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


# Convert time spec to seconds
def convert_time(t: Union[str, float, int]) -> float:
    r"""Convert a time specification to seconds

    This interprets strings such as ``"60s"``, ``"30m"``, ``"2h"``, or
    ``"1d"`` as 60 seconds, 30 minutes, 2 hours, and 1 day, respectively.
    Numbers and strings without a suffix are interpreted as seconds.

    :Call:
        >>> tsec = convert_time(t)
    :Inputs:
        *t*: :class:`float` | :class:`int` | :class:`str`
            Time as a number of seconds or a string with an optional
            suffix of ``"s"``, ``"m"``, ``"h"``, or ``"d"``
    :Outputs:
        *tsec*: :class:`float`
            Time converted to seconds
    :Raises:
        *OptdictValueError*: if *t* is not a valid time specification
    """
    # Check for string
    if isinstance(t, str):
        # Get last character as potential suffix
        suffix = t[-1:].lower()
        # Check for a valid suffix
        if suffix in TIME_SUFFIXES:
            # Strip suffix and convert
            return TIME_SUFFIXES[suffix] * float(t[:-1])
        elif not suffix.isalpha():
            # No suffix; interpret raw number as seconds
            return float(t)
        # Unrecognized letter suffix
        raise OptdictValueError(
            f"Unrecognized time suffix '{suffix}' in '{t}'; "
            "use one of 's', 'm', 'h', 'd'")
    # Already a number (or convertible)
    return float(t)


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
        "Interval": FLOAT_TYPES + INT_TYPES + (str,),
        "NCase": INT_TYPES,
        "StatusList": str,
        "Timeout": FLOAT_TYPES + INT_TYPES + (str,),
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
