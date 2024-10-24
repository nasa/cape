#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.cfdx.cli`: Command-line interface to ``cape`` (executable)
======================================================================

This module provides the :func:`main` function that is used by the
executable called ``cape``.

"""

# Standard library modules
import sys
from typing import Union

# CAPE modules
from .. import argread
from .. import convert1to2
from .. import text as textutils
from .cfdx_doc import CAPE_HELP
from .cntl import Cntl
from ..argread._vendor.kwparse import BOOL_TYPES, INT_TYPES


# Convert True -> 1 else txt -> int(txt)
def _true_int(txt: Union[bool, str]) -> int:
    # Check type
    if txt is True:
        return 1
    elif txt is False:
        return 0
    else:
        return int(txt)


# Argument settings for main run interface
class CapeRunArgs(argread.ArgReader):
    __slots__ = ()

    _arglist = (
        "cmd",
    )

    _optlist = (
        "I",
        "PASS",
        "FAIL",
        "apply",
        "c",
        "cons",
        "e",
        "extend",
        "f",
        "failed",
        "filter",
        "fm",
        "j",
        "ll",
        "marked",
        "n",
        "passed",
        "qsub",
        "re",
        "report",
        "start",
        "u",
        "unmark",
        "unmarked",
        "v",
        "x",
    )

    _optmap = {
        "ERROR": "FAIL",
        "aero": "fm",
        "check": "c",
        "exec": "e",
        "file": "f",
        "json": "f",
        "kill": "qdel",
        "regex": "re",
        "scancel": "qdel",
        "verbose": "v",
    }

    _opttypes = {
        "I": str,
        "apply": bool,
        "c": bool,
        "cmd": str,
        "e": str,
        "extend": int,
        "f": str,
        "filter": str,
        "fm": (bool, str),
        "j": bool,
        "ll": (bool, str),
        "n": int,
        "re": str,
        "report": (bool, str),
        "triqfm": (bool, str),
        "x": str
    }

    _optvals = {
        "cmd": (
            "start",
            "apply",
            "archive",
            "check"
            "clean",
            "extend",
            "extract-fm",
            "extract-ll",
            "extract-triqfm",
            "fail",
            "pass",
            "qdel",
            "rm",
            "skeleton",
            "unarchive",
            "unmark",
        )
    }

    _rawopttypes = {
        "extend": BOOL_TYPES + INT_TYPES,
    }

    _optconverters = {
        "extend": _true_int,
        "n": int,
    }

    _optlist_noval = (
        "PASS",
        "FAIL",
        "apply",
        "c",
        "j",
        "v",
        "unmark",
        "unmarked",
    )

    _help_optlist = (
        "h",
        "c",
        "n",
        "I",
        "re",
        "extend",
    )

    _help_opt = {
        "c": "check case(s) status",
        "e": "execute the command *EXEC*",
        "f": "use the JSON (or YAML) file *JSON*",
        "h": "print this help message and exit",
    }

    _help_optarg = {
        "e": "EXEC",
        "f": "JSON",
    }


# Primary interface
def main():
    r"""Main interface to ``pyfun``

    This is basically an interface to :func:`cape.cfdx.cntl.Cntl.cli`.

    :Call:
        >>> main()
    :Versions:
        * 2021-03-04 ``@ddalle``: Version 1.0
    """
    # Parse inputs
    a, kw = argread.readflagstar(sys.argv)

    # Check for a help flag
    if kw.get('h') or kw.get("help"):
        # Display help
        print(textutils.markdown(CAPE_HELP))
        return

    if kw.get("1to2"):
        print("Updating CAPE 1 -> 2")
        convert1to2.upgrade1to2()
        return

    # Get file name
    fname = kw.get('f', "cape.json")

    # Try to read it
    cntl = Cntl(fname)

    # Call the command-line interface
    cntl.cli(*a, **kw)


# Check if run as a script.
if __name__ == "__main__":
    main()

