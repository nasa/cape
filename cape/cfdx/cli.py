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
import os
import sys
from typing import Optional, Tuple, Union

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
IERR_OPT = 32

# Inferred commands from options
CMD_NAMES = {
    "batch": "batch",
    "c": "check",
    "PASS": "approve",
    "FAIL": "fail",
    "unmark": "unmark",
    "dezombie": "dezombie",
    "extend": "extend",
    "apply": "apply",
    "fm": "extract-fm",
    "ll": "extract-ll",
    "triqfm": "extract-triqfm",
    "report": "report",
    "check-db": "check-db",
    "e": "exec",
    "clean": "clean",
    "archive": "archive",
    "skeleton": "skeleton",
    "h": "help",
}


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
        "cons": "CONS",
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
        "h",
        "I",
        "cons",
        "f",
        "filter",
        "glob",
        "marked",
        "re",
        "unmarked",
        "x",
    )


# Settings for -c
class CfdxCheckArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-check"

    # Description
    _help_title = "Check status of one or more cases"

    # Additional options
    _optlist = (
        "c",
        "j",
    )


# Settings for --PASS
class CfdxApproveArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-approve"

    # Description
    _help_title = "Mark selected cases as complete"

    # Additional options
    _optlist = (
        "PASS",
    )


# Settings for --FAIL
class CfdxFailArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-fail"

    # Description
    _help_title = "Mark selected cases as ERRORs"

    # Additional options
    _optlist = (
        "FAIL",
    )


# Settings for --unmark
class CfdxUnmarkArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-unmark"

    # Description
    _help_title = "Remove PASS/ERROR markings for selected cases"

    # Additional options
    _optlist = (
        "unmark",
    )


# Settings for --extend
class CfdxExtendArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extend"

    # Description
    _help_title = "Extend unmarked cases"

    # Additional options
    _optlist = (
        "extend",
        "imax",
    )

    # Default values
    _rc = {
        "extend": 1,
    }


# Settings for --batch
class CfdxBatchArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-batch"

    # Description
    _help_title = "Submit CAPE command as a PBS/Slurm batch job"


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
        "approve",
        "archive",
        "batch",
        "check",
        "clean",
        "dezombie",
        "exec",
        "extend",
        "extract-fm",
        "extract-ll",
        "extract-triqfm",
        "check-db",
        "fail",
        "qdel",
        "rm",
        "skeleton",
        "unarchive",
        "unmark",
    )

    # Subparsers
    _cmdparsers = {
        "approve": CfdxApproveArgs,
        "batch": CfdxBatchArgs,
        "check": CfdxCheckArgs,
        "fail": CfdxFailArgs,
        "extend": CfdxExtendArgs,
        "unmark": CfdxUnmarkArgs,
    }

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
        for opt, cmdname in CMD_NAMES.items():
            if opt in self:
                return cmdname
        # Default is "run"
        return "run"


def read_cntl(cntl_cls: type, parser: CfdxArgReader) -> Cntl:
    r"""Read a CAPE run matrix control instance of appropriate class

    :Call:
        >>> cntl = read_cntl(cntl_cls, parser)
    :Inputs:
        *cntl_cls*: :class:`type`
            One of the CAPE :class:`cape.cfdx.cntl.Cntl` subclasses
        *parser*: :class:`CfdxArgReader`
            CLI parser instance
    :Outputs:
        *cntl*: *cntl_cls*
            CAPE run matrix control instance
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Get file name
    fname = parser.get_opt("f")
    # Instantiate
    return cntl_cls(fname)


def cape_approve(parser: CfdxArgReader, cntl_cls: type) -> int:
    r"""Run the ``cape --PASS`` command

    :Call:
        >>> ierr == cape_approve(parser, cntl_cls)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
        *cntl_cls*: :class:`type`
            CAPE run matrix control subclass to use
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.MarkPASS(**kw)
    # Return code
    return IERR_OK


def cape_c(parser: CfdxArgReader, cntl_cls: type) -> int:
    r"""Run the ``cape -c`` command

    :Call:
        >>> ierr == cape_c(parser, cntl_cls)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
        *cntl_cls*: :class:`type`
            CAPE run matrix control subclass to use
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.DisplayStatus(**kw)
    # Return code
    return IERR_OK


def cape_batch(parser: CfdxArgReader, cntl_cls: type) -> int:
    r"""Run the ``cape --batch`` command

    :Call:
        >>> ierr == cape_batch(parser, cntl_cls)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
        *cntl_cls*: :class:`type`
            CAPE run matrix control subclass to use
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-20 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Reconstruct command-line args
    argv = parser.reconstruct()
    # Remove ``-batch`` from command name
    cmdname = argv[0]
    if cmdname.endswith("-batch"):
        argv[0] = cmdname.rsplit('-', 1)[0]
    # Check for explicit executable
    pyexec = cntl.opts.get_PythonExec()
    if pyexec:
        # Full name of module: "cfdx" -> "cape.cfdx"
        argv[0] = f"cape.{argv[0]}"
        # Prepend python3 -m ...
        argv = [pyexec, '-m'] + argv
    # Remove recursive batch
    if "--batch" in argv:
        argv.remove("--batch")
    # Run the command
    cntl.run_batch(argv)
    # Return code
    return IERR_OK


def cape_fail(parser: CfdxArgReader, cntl_cls: type) -> int:
    r"""Run the ``cape --FAIL`` command

    :Call:
        >>> ierr == cape_fail(parser, cntl_cls)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
        *cntl_cls*: :class:`type`
            CAPE run matrix control subclass to use
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.MarkERROR(**kw)
    # Return code
    return IERR_OK


def cape_unmark(parser: CfdxArgReader, cntl_cls: type) -> int:
    r"""Run the ``cape --unmark`` command

    :Call:
        >>> ierr == cape_unmark(parser, cntl_cls)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
        *cntl_cls*: :class:`type`
            CAPE run matrix control subclass to use
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-20 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UnmarkCase(**kw)
    # Return code
    return IERR_OK


def read_cntl_kwargs(
        cntl_cls: type,
        parser: CfdxArgReader) -> Tuple[Cntl, dict]:
    r"""Read a CAPE run matrix control instance of appropriate class

    :Call:
        >>> cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    :Inputs:
        *cntl_cls*: :class:`type`
            One of the CAPE :class:`cape.cfdx.cntl.Cntl` subclasses
        *parser*: :class:`CfdxArgReader`
            CLI parser instance
    :Outputs:
        *cntl*: *cntl_cls*
            CAPE run matrix control instance
        *kw*: :class:`dict`
            Preprocessed keyword arguments
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Get file name
    fname = parser.get_opt("f")
    # Instantiate
    cntl = cntl_cls(fname)
    # Parse arguments
    kw = parser.get_kwargs()
    # Preprocess
    cntl.preprocess_kwargs(kw)
    # Output
    return cntl, kw


# Name -> Function
CMD_DICT = {
    "approve": cape_approve,
    "batch": cape_batch,
    "check": cape_c,
    "fail": cape_fail,
    "unmark": cape_unmark,
}


def main_template(
        cntl_cls: type,
        parser_cls: argread.ArgReader,
        argv: Optional[list] = None) -> int:
    # Create parser
    parser = parser_cls()
    # Use sys.argv if necessary
    argv = _get_argv(argv)
    # Identify subcommand
    try:
        cmdname, subparser = parser.fullparse(argv)
    except (NameError, ValueError, TypeError) as e:
        print("In command:\n")
        print("  " + " ".join(argv) + "\n")
        print({e.args[0]})
        return IERR_OPT
    # Help message
    if cmdname is None or cmdname == "help":
        print(compile_rst(parser.genr8_help()))
        return IERR_OK
    # Check for valid command name
    ierr = _help_frontdesk(cmdname, parser_cls)
    if ierr != IERR_OK:
        return ierr
    # Check for ``-h``
    if _help(subparser):
        return IERR_OK
    # Get function
    func = CMD_DICT.get(cmdname)
    # Call it
    if func:
        return func(subparser, cntl_cls)
    # For now, print the selected command
    print(cmdname)
    return IERR_OK


def main1(argv: Optional[list] = None) -> int:
    return main_template(Cntl, CfdxFrontDesk, argv)


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


def _get_argv(argv: Optional[list]) -> list:
    # Get sys.argv if needed
    argv = list(sys.argv) if argv is None else argv
    # Check for name of executable
    cmdname = argv[0]
    if cmdname.endswith("__main__.py"):
        # Get module name
        argv[0] = os.path.basename(os.path.dirname(cmdname))
    # Output
    return argv


def _help_frontdesk(cmdname: Optional[str], cls: type) -> int:
    r"""Display help message for front-desk parser, if appropriate"""
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


# Print help message
def _help(parser: CfdxArgReader) -> bool:
    r"""Generate help message for non-front-desk command

    :Call:
        >>> q = _help(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *q*: :class:`bool`
            Whether help message was displayed
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Check for help message
    if parser.get("h", False) and parser._cmdlist is None:
        # Print help message
        print(compile_rst(parser.genr8_help()))
        return True
    else:
        # No help
        return False

