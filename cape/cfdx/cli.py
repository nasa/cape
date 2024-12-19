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
class CfdxFrontDesk(argread.ArgReader):
    __slots__ = ()

    _name = "cape-cfdx"
    _help_title = "Control generic-solver run matrix"

    _optlist = (
        "I",
        "PASS",
        "FAIL",
        "apply",
        "archive",
        "c",
        "clean",
        "compile",
        "cons",
        "dezombie",
        "e",
        "extend",
        "f",
        "failed",
        "filter",
        "force",
        "fm",
        "imax",
        "incremental",
        "j",
        "ll",
        "marked",
        "n",
        "passed",
        "prompt",
        "pt",
        "q",
        "qsub",
        "re",
        "report",
        "restart",
        "rm",
        "skeleton",
        "start",
        "triqfm",
        "u",
        "unmark",
        "unmarked",
        "v",
        "x",
    )

    _optmap = {
        "ERROR": "FAIL",
        "aero": "fm",
        "approve": "PASS",
        "check": "c",
        "constraints": "cons",
        "exec": "e",
        "file": "f",
        "help": "h",
        "json": "f",
        "kill": "qdel",
        "queue": "q",
        "regex": "re",
        "scancel": "qdel",
        "verbose": "v",
    }

    _opttypes = {
        "I": str,
        "FAIL": bool,
        "PASS": bool,
        "apply": bool,
        "archive": bool,
        "auto": bool,
        "c": bool,
        "clean": bool,
        "compile": bool,
        "cons": str,
        "cmd": str,
        "dezombie": bool,
        "e": str,
        "extend": (bool, int),
        "f": str,
        "failed": bool,
        "filter": str,
        "force": bool,
        "fm": (bool, str),
        "imax": int,
        "incremental": bool,
        "j": bool,
        "ll": (bool, str),
        "marked": bool,
        "n": int,
        "passed": bool,
        "prompt": bool,
        "pt": (bool, str),
        "q": bool,
        "qsub": bool,
        "re": str,
        "report": (bool, str),
        "restart": bool,
        "rm": bool,
        "skeleton": bool,
        "start": bool,
        "triqfm": (bool, str),
        "u": str,
        "unmark": bool,
        "unmarked": bool,
        "v": bool,
        "x": str,
    }

    _cmdlist = (
        "start",
        "apply",
        "archive",
        "check"
        "clean",
        "dezombie",
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

    _help_cmd = {
        "start": "Start/submit case(s)",
    }

    _rawopttypes = {
        "extend": BOOL_TYPES + INT_TYPES,
    }

    _rc = {
        "extend": 1,
        "restart": True,
        "start": True,
    }

    _optconverters = {
        "extend": _true_int,
        "imax": int,
        "n": int,
    }

    _optlist_noval = (
        "PASS",
        "FAIL",
        "apply",
        "auto",
        "c",
        "compile",
        "dezombie",
        "force",
        "incremental",
        "j",
        "marked",
        "prompt",
        "restart",
        "rm",
        "start",
        "unmark",
        "unmarked",
        "v",
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
        "marked",
        "unmarked",
        "rm",
        "apply",
        "extend",
        "PASS",
        "FAIL",
        "unmark",
        "fm",
        "ll",
        "triqfm",
        "dezombie",
        "e",
        "restart",
        "start",
        "clean",
        "archive",
        "skeleton",
        "q",
        "u",
        "x",
    )

    _help_opt = {
        "FAIL": "Mark case(s) as ERRORs",
        "I": "Specific case indices, e.g. ``-I 4:8,12``",
        "PASS": "Marke case(s) as PASS",
        "apply": "Apply current JSON settings to existing case(s)",
        "archive": "Archive files from case(s) and delete extra files",
        "auto": "Ignore *RunControl* > *NJob* if set",
        "c": "Check and display case(s) status",
        "clean": "Remove files not necessary for running and not archived",
        "compile": "Create images for report but don't compile PDF",
        "cons": 'Constraints on run matrix keys, e.g. ``"mach>1.0"``',
        "dezombie": "Clean up ZOMBIE cases, RUNNING but no recent file mods",
        "e": "Execute the command *EXEC*",
        "extend": "Extend case(s) by *N_EXT* copies of last phase",
        "f": "Use the JSON (or YAML) file *JSON*",
        "filter": "Limit to cases containing the string *TXT*",
        "fm": "Extract force & moment data [comps matching *PAT*] for case(s)",
        "force": "Update report and ignore subfigure cache",
        "glob": "Limit to cases whose name matches the filename pattern *PAT*",
        "h": "Print this help message and exit",
        "imax": "Do not extend a case beyond iteration *M*",
        "incremental": "Run case for one phase [or stop after *STOP_PHASE*]",
        "j": "List PBS/Slurm job ID in ``-c`` output",
        "kill": "Remove jobs from the queue and stop them",
        "ll": "Extract line load data [comps matching *PAT*] for case(s)",
        "marked": "Show only cases marked either PASS or ERROR",
        "n": "Submit at most *N* cases",
        "pt": "Extract surf point sensors [comps matching *PAT*] for case(s)",
        "prompt": "Don't ask for confirmation when deleting cases w/o iters",
        "q": "Submit to PBS/Slurm queue *QUEUE*, overrides value in JSON file",
        "re": "Limit to cases containing regular expression *REGEX*",
        "report": "Generate the report *RP* or the first in the list",
        "restart": "When submitting new jobs, only submit new cases",
        "skeleton": "Delete most files from indicaded PASSED cases",
        "rm": "Remove indicated cases",
        "start": "Set up but do not start (or submit) cases",
        "triqfm": "Extract triq F&M data [comps matching *PAT*] for case(s)",
        "u": "Pretend to be user *UID*",
        "unmark": "Remove PASS/ERROR marking for case(s)",
        "unmarked": "Show cases with no PASS/ERROR markings",
        "x": "Execute Python script *PYSCRIPT* after reading JSON",
    }

    _help_optarg = {
        "I": "INDS",
        "e": "EXEC",
        "extend": "[N_EXT]",
        "f": "JSON",
        "filter": "TXT",
        "fm": "[PAT]",
        "glob": "PAT",
        "imax": "M",
        "incremental": "[STOP_PHASE]",
        "ll": "[PAT]",
        "n": "N",
        "pt": "[PAT]",
        "q": "QUEUE",
        "re": "REGEX",
        "report": "[RP]",
        "triqfm": "[PAT]",
        "u": "UID",
        "x": "PYSCRIPT",
    }

    _help_opt_negative = (
        "auto",
        "compile",
        "prompt",
        "restart",
        "start",
    )


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

