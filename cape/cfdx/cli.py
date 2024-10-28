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

    _name = "cape-cfdx"
    _help_title = "Control generic-solver run matrix"

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
        "constraints": "cons",
        "exec": "e",
        "file": "f",
        "help": "h",
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

    _rc = {
        "extend": 1,
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
        "f",
        "c",
        "j",
        "n",
        "I",
        "cons",
        "re",
        "filter",
        "e",
        "extend",
        "u",
        "x",
    )

    _help_opt = {
        "I": "specific case indices, e.g. ``-I 4:8,12``",
        "apply": "apply current JSON settings to existing case(s)",
        "c": "check case(s) status",
        "cons": "comma-sep constraints on run matrix keys, e.g. ``mach>1.0``",
        "cmd": "name of sub-command to run",
        "e": "execute the command *EXEC*",
        "extend": "extend case(s) by *N_EXT* copies of last phase",
        "f": "use the JSON (or YAML) file *JSON*",
        "filter": "limit to cases containing specific text",
        "fm": "extract force & moment data [comps matching *PAT*] for case(s)",
        "h": "print this help message and exit",
        "j": "list PBS/Slurm job ID in ``-c`` output",
        "ll": "extract line load data [comps matching *PAT*] for case(s)",
        "n": "maximum number of jobs to submit/start",
        "pt": "extract surf point sensors [comps matching *PAT*] for case(s)",
        "re": "limit to cases containing regular expression *REGEX*",
        "triqfm": "extract triq F&M data [comps matching *PAT*] for case(s)",
        "x": "execute script *PYSCRIPT* after reading JSON",
    }

    _help_optarg = {
        "I": "INDS",
        "e": "EXEC",
        "extend": "[N_EXT]",
        "f": "JSON",
        "fm": "[PAT]",
        "ll": "[PAT]",
        "pt": "[PAT]",
        "re": "REGEX",
        "triqfm": "[PAT]",
        "u": "UID",
        "x": "PYSCRIPT",
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

