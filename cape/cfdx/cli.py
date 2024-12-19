#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.cfdx.cli`: Command-line interface to ``cape`` (executable)
======================================================================

This module provides the :func:`main` function that is used by the
executable called ``cape``.

"""

# Standard library modules
import difflib
import sys
from typing import Optional, Union

# CAPE modules
from .. import argread
from .. import convert1to2
from .. import text as textutils
from .cfdx_doc import CAPE_HELP
from .cntl import Cntl
from ..argread.clitext import compile_rst
from ..argread._vendor.kwparse import BOOL_TYPES, INT_TYPES


# Constants
IERR_OK = 0
IERR_CMD = 16


# Convert True -> 1 else txt -> int(txt)
def _true_int(txt: Union[bool, str]) -> int:
    # Check type
    if txt is True:
        return 1
    elif txt is False:
        return 0
    else:
        return int(txt)


# Common argument settings
class CfdxArgReader(argread.ArgReader):
    # No attributes
    __slots__ = ()

    # Common aliases
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

    # Option types
    _opttypes = {
        "I": str,
        "FAIL": bool,
        "PASS": bool,
        "apply": bool,
        "archive": bool,
        "auto": bool,
        "batch": bool,
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

    # Allowed types prior to conversion
    _rawopttypes = {
        "extend": BOOL_TYPES + INT_TYPES,
    }

    # Default values
    _rc = {
        "extend": 1,
        "restart": True,
        "start": True,
    }

    # Conversion functions
    _optconverters = {
        "extend": _true_int,
        "imax": int,
        "n": int,
    }

    # List of options that cannot take a "value"
    _optlist_noval = (
        "PASS",
        "FAIL",
        "apply",
        "auto",
        "batch",
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

    # Description of each option
    _help_opt = {
        "FAIL": "Mark case(s) as ERRORs",
        "I": "Specific case indices, e.g. ``-I 4:8,12``",
        "PASS": "Marke case(s) as PASS",
        "apply": "Apply current JSON settings to existing case(s)",
        "archive": "Archive files from case(s) and delete extra files",
        "auto": "Ignore *RunControl* > *NJob* if set",
        "batch": "Submit PBS/Slurm job and run this command there",
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

    # Name for value of select options in option descriptions
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

    # List of options that should be shown as negative in help
    _help_opt_negative = (
        "auto",
        "compile",
        "prompt",
        "restart",
        "start",
    )


# Settings for subsect commands
class _CfdxSubsetArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # List of available options
    _optlist = (
        "I",
        "cons",
        "filter",
        "glob",
        "marked",
        "re",
        "unmarked",
    )


# Argument settings for main run interface
class CfdxFrontDesk(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of executable
    _name = "cape-cfdx"

    # Description of executable
    _help_title = "Control generic-solver run matrix"

    # List of available options
    _optlist = (
        "FAIL",
        "I",
        "PASS",
        "apply",
        "archive",
        "auto",
        "batch",
        "c",
        "clean",
        "compile",
        "cons",
        "dezombie",
        "e",
        "extend",
        "f",
        "filter",
        "fm",
        "force",
        "glob",
        "h",
        "imax",
        "incremental",
        "j",
        "kill",
        "ll",
        "marked",
        "n",
        "pt",
        "prompt",
        "q",
        "re",
        "report",
        "restart",
        "skeleton",
        "rm",
        "start",
        "triqfm",
        "u",
        "unmark",
        "unmarked",
        "x",
    )

    # List of sub-commands
    _cmdlist = (
        "help",
        "run",
        "run-case",
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

    # Description of sub-commands
    _help_cmd = {
        "help": "Display help message and exit",
        "run": "Start/submit case(s)",
        "run-case": "Run case in current folder",
    }

    # List of options for --help
    _help_optlist = (
        "h",
        "f",
        "j",
        "n",
        "I",
        "cons",
        "re",
        "filter",
        "marked",
        "unmarked",
        "batch",
        "e",
        "restart",
        "start",
        "q",
        "u",
        "x",
    )

    # Decide on sub-command if none specified
    def infer_cmdname(self) -> str:
        # Check for various options
        if self.get("h"):
            # Overall help (front-desk)
            return "help"
        elif self.get("c"):
            # Check takes precedence
            return "check"
        else:
            # Default is to start cases
            return "run"


# Print help message for front-desk
def _help_frontdesk(cmdname: Optional[str], cls: type) -> int:
    # Check for null commands
    if cmdname is None:
        print(compile_rst(cls().genr8_help()))
        return IERR_OK
    # Check if command was recognized
    if cmdname not in cls._cmdlist:
        # Get closest matches
        close = difflib.get_close_matches(
            cmdname, cls._cmdlist, n=4, cutoff=0.3)
        # Use all if no matches
        close = close if close else cls._cmdlist
        # Generate list as text
        matches = " | ".join(close)
        # Display them
        print(f"Unexpected '{cls._name}' command '{cmdname}'")
        print(f"Closest matches: {matches}")
        return IERR_CMD
    # No problems
    return IERR_OK


# New interface
def main1(argv: Optional[list] = None) -> int:
    # Create parser
    parser = CfdxFrontDesk()
    # Use sys.argv if necesary
    argv = sys.argv if argv is None else argv
    # Identify subcommand
    cmdname, subparser = parser.fullparse(argv)
    # Help message
    if cmdname is None or cmdname == "help":
        print(compile_rst(parser.genr8_help()))
        return IERR_OK
    # Check for valid command name
    ierr = _help_frontdesk(cmdname, CfdxFrontDesk)
    if ierr != IERR_OK:
        return ierr


# Primary interface
def main():
    r"""Main interface to ``cape-cfdx``

    This is basically an interface to :func:`cape.cfdx.cntl.Cntl.cli`.

    :Call:
        >>> main()
    :Versions:
        * 2021-03-04 ``@ddalle``: v1.0
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

