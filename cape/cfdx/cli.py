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
from .casecntl import CaseRunner
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
    "dbpyfunc": "extract-pyfunc",
    "fm": "extract-fm",
    "ll": "extract-ll",
    "prop": "extract-prop",
    "pt": "extract-triqpt",
    "triqfm": "extract-triqfm",
    "ts": "extract-timeseries",
    "report": "report",
    "check-db": "check-db",
    "e": "exec",
    "clean": "clean",
    "archive": "archive",
    "skeleton": "skeleton",
    "report": "report",
    "rm": "rm",
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
    __slots__ = (
        "cntl_cls",
        "runner_cls",
    )

    # Common aliases
    _optmap = {
        "ERROR": "FAIL",
        "aero": "fm",
        "approve": "PASS",
        "check": "c",
        "checkDB": "check-db",
        "checkFM": "check-fm",
        "checkLL": "check-ll",
        "checkTriqFM": "check-triqfm",
        "constraints": "cons",
        "exec": "e",
        "fail": "FAIL",
        "file": "f",
        "force": "F",
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
        "check-db": bool,
        "check-fm": bool,
        "check-ll": bool,
        "check-triqfm": bool,
        "clean": bool,
        "compile": bool,
        "cons": str,
        "cmd": str,
        "dbpyfunc": (bool, str),
        "delete": bool,
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
        "prop": (bool, str),
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
        "ts": (bool, str),
        "u": str,
        "unmark": bool,
        "unmarked": bool,
        "v": bool,
        "x": str,
    }

    # Allowed types prior to conversion
    _rawopttypes = {
        "extend": BOOL_TYPES + INT_TYPES + (str,),
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
        "delete",
        "dezombie",
        "force",
        "incremental",
        "j",
        "marked",
        "prompt",
        "qsub",
        "qdel",
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
        "check-db": "Check completion of all databook products",
        "check-fm": "Check completion of force & moment components",
        "check-ll": "Check completion of line load components",
        "check-triqfm": "Check completion of patch load (triqfm) components",
        "clean": "Remove files not necessary for running and not archived",
        "compile": "Create images for report but don't compile PDF",
        "cons": 'Constraints on run matrix keys, e.g. ``"mach>1.0"``',
        "dbpyfunc": "Extract scalar data from custom Python function",
        "delete": "Delete DataBook entries instead of adding new ones",
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
        "prop": "Extract scalar properties [comps matching *PAT*]",
        "q": "Submit to PBS/Slurm queue *QUEUE*, overrides value in JSON file",
        "qsub": "Don't submit PBS/Slurm jobs even if otherwise specified",
        "re": "Limit to cases containing regular expression *REGEX*",
        "report": "Generate the report *RP* or the first in the list",
        "restart": "When submitting new jobs, only submit new cases",
        "skeleton": "Delete most files from indicaded PASSED cases",
        "rm": "Remove indicated cases",
        "start": "Set up but do not start (or submit) cases",
        "triqfm": "Extract triq F&M data [comps matching *PAT*] for case(s)",
        "ts": "Extract time-series data [comps matching *PAT*]",
        "u": "Pretend to be user *UID*",
        "unmark": "Remove PASS/ERROR marking for case(s)",
        "unmarked": "Show cases with no PASS/ERROR markings",
        "x": "Execute Python script *PYSCRIPT* after reading JSON",
    }

    # Name for value of select options in option descriptions
    _help_optarg = {
        "I": "INDS",
        "cons": "CONS",
        "dbpyfunc": "[PAT]",
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
        "prop": "[PAT]",
        "pt": "[PAT]",
        "q": "QUEUE",
        "re": "REGEX",
        "report": "[RP]",
        "triqfm": "[PAT]",
        "ts": "[PAT]",
        "u": "UID",
        "x": "PYSCRIPT",
    }

    # List of options that should be shown as negative in help
    _help_opt_negative = (
        "auto",
        "compile",
        "prompt",
        "qsub",
        "restart",
        "start",
    )


# Settings for subset commands
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


# Settings for databook commands
class _CfdxExtractArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "delete",
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
        "u",
    )


# Settings for --archive
class CfdxArchiveArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-archive"

    # Description
    _help_title = "Archive cases; delete files not needed for post-processing"

    # Additional options
    _optlist = (
        "archive",
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


# Settings for --batch
class CfdxBatchArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-batch"

    # Description
    _help_title = "Submit CAPE command as a PBS/Slurm batch job"


# Settings for --check-db
class CfdxCheckDBArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-check-db"

    # Description
    _help_title = "Check completion of all databook components"

    # Additional options
    _optlist = (
        "check-db",
        "check-fm",
        "check-ll",
        "check-triqfm",
    )

    # Default values
    _rc = {
        "check-db": True,
    }


# Settings for --check-fm
class CfdxCheckFMArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-check-fm"

    # Description
    _help_title = "Check completion of all force & moment components"

    # Additional options
    _optlist = (
        "check-fm",
    )

    # Default values
    _rc = {
        "check-fm": True,
    }


# Settings for --check-ll
class CfdxCheckLLArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-check-ll"

    # Description
    _help_title = "Check completion of all line load components"

    # Additional options
    _optlist = (
        "check-ll",
    )

    # Default values
    _rc = {
        "check-ll": True,
    }


# Settings for --check-triqfm
class CfdxCheckTriqFMArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-check-triqfm"

    # Description
    _help_title = "Check completion of TriqFM components"

    # Additional options
    _optlist = (
        "check-triqfm",
    )

    # Default values
    _rc = {
        "check-triqfm": True,
    }


# Settings for --clean
class CfdxCleanArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-clean"

    # Description
    _help_title = "Remove extra files not necessary for running a case"

    # Additional options
    _optlist = (
        "clean",
    )


# Settings for --dezombie
class CfdxDezombieArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-dezombie"

    # Description
    _help_title = "Delete job and clean-up stalled cases (aka 'zombie' cases)"

    # Additional options
    _optlist = (
        "dezombie",
    )


# Settings for -e
class CfdxExecArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-exec"

    # Description
    _help_title = "Run a shell command in folder of case(s)"

    # Additional options
    _optlist = (
        "e",
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
        "qsub",
    )

    # Default values
    _rc = {
        "extend": 1,
    }


# Settings for --fm
class CfdxExtractFMArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-fm"

    # Description
    _help_title = "Extract averaged force & moment results"

    # Additional options
    _optlist = (
        "fm",
    )

    # Positional parameters
    _arglist = (
        "fm",
    )


# Settings for --ll
class CfdxExtractLLArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-ll"

    # Description
    _help_title = "Compute and extract line load results"

    # Additional options
    _optlist = (
        "ll",
    )

    # Positional parameters
    _arglist = (
        "ll",
    )


# Settings for --prop
class CfdxExtractPropArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-prop"

    # Description
    _help_title = "Extract other scalar results"

    # Additional options
    _optlist = (
        "prop",
    )

    # Positional parameters
    _arglist = (
        "prop",
    )


# Settings for --dbpyfunc
class CfdxExtractPyFuncArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-pyfunc"

    # Description
    _help_title = "Extract data from custom Python functions"

    # Additional options
    _optlist = (
        "dbpyfunc",
    )

    # Positional parameters
    _arglist = (
        "dbpyfunc",
    )


# Settings for --ts
class CfdxExtractTimeSeriesArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-timeseries"

    # Description
    _help_title = "Extract time series data"

    # Additional options
    _optlist = (
        "ts",
    )

    # Positional parameters
    _arglist = (
        "ts",
    )


# Settings for --triqfm
class CfdxExtractTriqFMArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-triqfm"

    # Description
    _help_title = "Extract post-processed patch loads"

    # Additional options
    _optlist = (
        "triqfm",
    )

    # Positional parameters
    _arglist = (
        "triqfm",
    )


# Settings for --pt
class CfdxExtractTriqPTArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-triqpt"

    # Description
    _help_title = "Collect post-processed point sensor data"

    # Additional options
    _optlist = (
        "pt",
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


# Settings for --qdel
class CfdxQdelArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-qdel"

    # Description
    _help_title = "Delete PBS/Slurm job of case(s)"

    # Additional options
    _optlist = (
        "qdel",
    )

    # Defaults
    _rc = {
        "qdel": True,
    }


# Settings for --report
class CfdxReportArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-report"

    # Description
    _help_title = "Update automated PDF reports"

    # Additional options
    _optlist = (
        "compile",
        "force",
        "report",
        "rm",
    )

    # Alternate descriptions
    _help_opt = {
        "rm": "Remove report figures instead of updating report",
    }


# Settings for --rm
class CfdxRemoveCasesArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-rm"

    # Description
    _help_title = "Delete entire case folders"

    # Additional options
    _optlist = (
        "rm",
        "prompt",
    )

    # Defaults
    _rc = {
        "rm": True,
    }


# Settings for run
class CfdxRunArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-run"

    # Description
    _help_title = "Run case in current folder"

    # Options
    _optlist = (
        "h",
    )


# Settings for --skeleton
class CfdxSkeletonArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-skeleton"

    # Description
    _help_title = "Clean up case folder; leave only key files"

    # Additional options
    _optlist = (
        "skeleton",
    )


# Settings for -n
class CfdxStartArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-start"

    # Description
    _help_title = "Setup, start, and/or submit cases"

    # Additional options
    _optlist = (
        "n",
        "j",
        "q",
        "qsub",
        "u",
        "start",
    )


# Settings for --unarchive
class CfdxUnarchiveArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-unarchive"

    # Description
    _help_title = "Expand files from archive"

    # Additional options
    _optlist = (
        "unarchive",
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


# Argument settings for main run interface
class CfdxFrontDesk(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of executable
    _name = "cape-cfdx"

    # Description of executable
    _help_title = "Control generic-solver run matrix"

    # Special classes
    _cntl_cls = Cntl
    _runner_cls = CaseRunner

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
        "check-db",
        "check-fm",
        "check-ll",
        "check-triqfm",
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
        "qsub",
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
        "start",
        "check",
        "apply",
        "approve",
        "archive",
        "batch",
        "check-db",
        "check-fm",
        "check-ll",
        "check-triqfm",
        "clean",
        "dezombie",
        "exec",
        "extend",
        "extract-fm",
        "extract-ll",
        "extract-pyfunc",
        "extract-prop",
        "extract-timeseries",
        "extract-triqfm",
        "extract-triqpt",
        "fail",
        "qdel",
        "report",
        "rm",
        "skeleton",
        "unarchive",
        "unmark",
    )

    # Alternate command names
    _cmdmap = {
        "c": "check",
        "e": "exec",
        "error": "fail",
        "mark-error": "fail",
        "mark-failure": "fail",
        "mark-pass": "approve",
        "pass": "approve",
        "r": "run",
        "qsub": "start",
        "submit": "start",
    }

    # Subparsers
    _cmdparsers = {
        "archive": CfdxArchiveArgs,
        "approve": CfdxApproveArgs,
        "batch": CfdxBatchArgs,
        "check": CfdxCheckArgs,
        "check-db": CfdxCheckDBArgs,
        "check-fm": CfdxCheckFMArgs,
        "check-ll": CfdxCheckLLArgs,
        "check-triqfm": CfdxCheckTriqFMArgs,
        "clean": CfdxCleanArgs,
        "dezombie": CfdxDezombieArgs,
        "exec": CfdxExecArgs,
        "extend": CfdxExtendArgs,
        "extract-fm": CfdxExtractFMArgs,
        "extract-ll": CfdxExtractLLArgs,
        "extract-pyfunc": CfdxExtractPyFuncArgs,
        "extract-prop": CfdxExtractPropArgs,
        "extract-timeseries": CfdxExtractTimeSeriesArgs,
        "extract-triqfm": CfdxExtractTriqFMArgs,
        "extract-triqpt": CfdxExtractTriqPTArgs,
        "fail": CfdxFailArgs,
        "qdel": CfdxQdelArgs,
        "report": CfdxReportArgs,
        "rm": CfdxRemoveCasesArgs,
        "run": CfdxRunArgs,
        "start": CfdxStartArgs,
        "skeleton": CfdxSkeletonArgs,
        "unarchive": CfdxUnarchiveArgs,
        "unmark": CfdxUnmarkArgs,
    }

    # Description of sub-commands
    _help_cmd = {
        "help": "Display help message and exit",
        "batch": "Resubmit this command as a PBS/Slurm job",
        "check": "Check status of case(s)",
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
        return "start"


def cape_archive(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --archive`` command

    :Call:
        >>> ierr == cape_archive(parser, cntl_cls)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.ArchiveCases(**kw)
    # Return code
    return IERR_OK


def cape_approve(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --PASS`` command

    :Call:
        >>> ierr == cape_approve(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.MarkPASS(**kw)
    # Return code
    return IERR_OK


def cape_batch(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --batch`` command

    :Call:
        >>> ierr == cape_batch(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-20 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, _ = read_cntl_kwargs(cntl_cls, parser)
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


def cape_c(parser: CfdxArgReader) -> int:
    r"""Run the ``cape -c`` command

    :Call:
        >>> ierr == cape_c(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.DisplayStatus(**kw)
    # Return code
    return IERR_OK


def cape_check_db(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --check-db`` command

    :Call:
        >>> ierr == cape_check_db(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2025-01-02 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.CheckFM(**kw)
    cntl.CheckLL(**kw)
    cntl.CheckTriqFM(**kw)
    # Return code
    return IERR_OK


def cape_check_fm(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --check-fm`` command

    :Call:
        >>> ierr == cape_check_fm(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2025-01-02 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.CheckFM(**kw)
    # Return code
    return IERR_OK


def cape_check_ll(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --check-ll`` command

    :Call:
        >>> ierr == cape_check_ll(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2025-01-02 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.CheckLL(**kw)
    # Return code
    return IERR_OK


def cape_check_triqfm(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --check-triqfm`` command

    :Call:
        >>> ierr == cape_check_triqfm(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2025-01-02 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.CheckTriqFM(**kw)
    # Return code
    return IERR_OK


def cape_clean(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --clean`` command

    :Call:
        >>> ierr == cape_clean(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.CleanCases(**kw)
    # Return code
    return IERR_OK


def cape_dezombie(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --dezombie`` command

    :Call:
        >>> ierr == cape_dezombie(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.Dezombie(**kw)
    # Return code
    return IERR_OK


def cape_exec(parser: CfdxArgReader) -> int:
    r"""Run the ``cape -e`` command

    :Call:
        >>> ierr == cape_exec(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.ExecScript(**kw)
    # Return code
    return IERR_OK


def cape_extend(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --extend`` command

    :Call:
        >>> ierr == cape_exec(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.ExtendCases(**kw)
    # Return code
    return IERR_OK


def cape_extract_fm(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --fm`` command

    :Call:
        >>> ierr == cape_extract_fm(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UpdateFM(**kw)
    # Return code
    return IERR_OK


def cape_extract_ll(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --ll`` command

    :Call:
        >>> ierr == cape_extract_ll(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UpdateLL(**kw)
    # Return code
    return IERR_OK


def cape_extract_prop(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --prop`` command

    :Call:
        >>> ierr == cape_extract_prop(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UpdateCaseProp(**kw)
    # Return code
    return IERR_OK


def cape_extract_pyfunc(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --dbpyfunc`` command

    :Call:
        >>> ierr == cape_extract_pyfunc(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UpdateDBPyFunc(**kw)
    # Return code
    return IERR_OK


def cape_extract_timeseries(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --ts`` command

    :Call:
        >>> ierr == cape_extract_timeseries(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UpdateTS(**kw)
    # Return code
    return IERR_OK


def cape_extract_triqfm(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --triqfm`` command

    :Call:
        >>> ierr == cape_extract_triqfm(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UpdateTriqFM(**kw)
    # Return code
    return IERR_OK


def cape_extract_triqpt(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --pt`` command

    :Call:
        >>> ierr == cape_extract_triqpt(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UpdateTriqPoint(**kw)
    # Return code
    return IERR_OK


def cape_fail(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --FAIL`` command

    :Call:
        >>> ierr == cape_fail(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.MarkERROR(**kw)
    # Return code
    return IERR_OK


def cape_qdel(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --qdel`` command to stop PBS/Slurm cases

    :Call:
        >>> ierr == cape_qdel(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-30 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.SubmitJobs(**kw)
    # Return code
    return IERR_OK


def cape_report(parser: CfdxArgReader) -> int:
    r"""Run the ``cape`` command to submit cases

    :Call:
        >>> ierr == cape_start(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-30 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Get name of report
    reportname = kw.get("report")
    # Use first report if no name given
    if not isinstance(reportname, str):
        reportname = cntl.opts.get_ReportList()[0]
    # Read the report
    report = cntl.ReadReport(reportname)
    # Check for force-update
    report.force_update = kw.get("force", False)
    # Check if asking to delete figures
    if kw.get("rm", False):
        # Remove the case(s) dir(s)
        report.RemoveCases(**kw)
    else:
        # Update report
        report.UpdateReport(**kw)
    # Return code
    return IERR_OK


def cape_rm(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --rm`` command to delete cases

    :Call:
        >>> ierr == cape_start(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.SubmitJobs(**kw)
    # Return code
    return IERR_OK


def cape_run(parser: CfdxArgReader) -> int:
    r"""Run the ``cape run`` command to run case in current folder

    :Call:
        >>> ierr == cape_start(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-30 ``@ddalle``: v1.0
    """
    # Get CaseRunner class
    runner_cls = getattr(parser, "runner_cls", CaseRunner)
    # Read instance
    runner, _ = read_runner_kwargs(runner_cls, parser)
    # Run the case
    runner.run()
    # Return code
    return IERR_OK


def cape_skeleton(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --skeleton`` command

    :Call:
        >>> ierr == cape_skeleton(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.SkeletonCases(**kw)
    # Return code
    return IERR_OK


def cape_start(parser: CfdxArgReader) -> int:
    r"""Run the ``cape`` command to submit cases

    :Call:
        >>> ierr == cape_start(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.SubmitJobs(**kw)
    # Return code
    return IERR_OK


def cape_unarchive(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --unarchive`` command

    :Call:
        >>> ierr == cape_unacrhive(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-29 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
    # Read instance
    cntl, kw = read_cntl_kwargs(cntl_cls, parser)
    # Run the command
    cntl.UnarchiveCases(**kw)
    # Return code
    return IERR_OK


def cape_unmark(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --unmark`` command

    :Call:
        >>> ierr == cape_unmark(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-20 ``@ddalle``: v1.0
    """
    # Get Cntl class
    cntl_cls = getattr(parser, "cntl_cls", Cntl)
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


def read_runner_kwargs(
        runner_cls: type,
        parser: CfdxArgReader) -> Tuple[CaseRunner, dict]:
    r"""Read a CAPE case runner instance to interact with CFD in ``PWD``

    :Call:
        >>> runner = read_runner(runner_cls, parser)
    :Inputs:
        *runner_cls*: :class:`type`
            Subclass of :class:`cape.cfdx.casecntl.CaseRunner`
        *parser*: :class:`CfdxArgReader`
            CLI parser instance
    :Outputs:
        *runner*: *runner_cls*
            Case runner instance to control case in current folder
        *kw*: :class:`dict`
            Preprocessed keyword arguments
    :Versions:
        * 2024-12-30 ``@ddalle``: v1.0
    """
    # Instantiate
    runner = runner_cls()
    # Parse arguments
    kw = parser.get_kwargs()
    # Output
    return runner, kw


# Name -> Function
CMD_DICT = {
    "archive": cape_archive,
    "approve": cape_approve,
    "batch": cape_batch,
    "check": cape_c,
    "check-db": cape_check_db,
    "check-fm": cape_check_fm,
    "check-ll": cape_check_ll,
    "check-triqfm": cape_check_triqfm,
    "clean": cape_clean,
    "dezombie": cape_dezombie,
    "exec": cape_exec,
    "extend": cape_extend,
    "extract-fm": cape_extract_fm,
    "extract-ll": cape_extract_ll,
    "extract-prop": cape_extract_prop,
    "extract-pyfunc": cape_extract_pyfunc,
    "extract-timeseries": cape_extract_timeseries,
    "extract-triqfm": cape_extract_triqfm,
    "extract-triqpt": cape_extract_triqpt,
    "fail": cape_fail,
    "qdel": cape_qdel,
    "report": cape_report,
    "rm": cape_rm,
    "run": cape_run,
    "skeleton": cape_skeleton,
    "start": cape_start,
    "unarchive": cape_unarchive,
    "unmark": cape_unmark,
}


def main_template(
        parser_cls: CfdxFrontDesk,
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
    # Set Cntl/CaseRunner classes for this solver
    subparser.cntl_cls = parser_cls._cntl_cls
    subparser.runner_cls = parser_cls._runner_cls
    # Get function
    func = CMD_DICT.get(cmdname)
    # Call it
    if func:
        return func(subparser)
    # For now, print the selected command
    return IERR_OK


def main1(argv: Optional[list] = None) -> int:
    return main_template(CfdxFrontDesk, argv)


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

