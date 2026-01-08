r"""
:mod:`cape.cfdx.cli`: Command-line interface to ``cape`` (executable)
======================================================================

This module provides the :func:`main` function that is used by the
executable called ``cape``.

"""

# Standard library modules
import importlib
import os
import sys
from typing import Optional, Union

# CAPE modules
from . import manage
from .. import argread
from .. import convert1to2
from ..argread import BOOL_TYPES, INT_TYPES


# Constants
IERR_OK = 0
IERR_CMD = 16
IERR_OPT = 32

# Inferred commands from options
CMD_NAMES = {
    "1to2": "1to2",
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
    "iter-fm": "extract-iter-fm",
    "ll": "extract-ll",
    "prop": "extract-prop",
    "surfcp": "extract-surfcp",
    "pt": "extract-triqpt",
    "triqfm": "extract-triqfm",
    "ts": "extract-timeseries",
    "report": "report",
    "check-db": "check-db",
    "e": "exec",
    "qdel": "qdel",
    "clean": "clean",
    "archive": "archive",
    "skeleton": "skeleton",
    "unarchive": "unarchive",
    "report": "report",
    "rm": "rm",
    "check-db": "check-db",
    "check-fm": "check-fm",
    "check-ll": "check-ll",
    "check-triqfm": "check-triqfm",
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
        "cntl_mod",
        "casecntl_mod",
        "cntl_cls",
        "runner_cls",
    )

    # Common aliases
    _optmap = {
        "ERROR": "FAIL",
        "F": "force",
        "add-col": "add-cols",
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
        "help": "h",
        "hide": "hide-cols",
        "iterfm": "iter-fm",
        "json": "f",
        "kill": "qdel",
        "minsize": "cutoff",
        "pattern": "pat",
        "queue": "q",
        "regex": "re",
        "scancel": "qdel",
        "verbose": "v",
    }

    # Option types
    _opttypes = {
        "1to2": bool,
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
        "cutoff": str,
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
        "iter-fm": (bool, str),
        "j": bool,
        "ll": (bool, str),
        "marked": bool,
        "n": int,
        "passed": bool,
        "pat": str,
        "prompt": bool,
        "prop": (bool, str),
        "pt": (bool, str),
        "q": bool,
        "qsub": bool,
        "re": str,
        "report": (bool, str),
        "restart": bool,
        "rm": bool,
        "surfcp": (bool, str),
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
        "1to2",
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
        "1to2": "Convert Python modules in current folder from CAPE 1 to 2",
        "FAIL": "Mark case(s) as ERRORs",
        "I": "Specific case indices, e.g. ``-I 4:8,12``",
        "PASS": "Mark case(s) as PASS",
        "add-cols": "Additional columns to show in run matrix status table",
        "add-counters": "Additional keys to show totals after run mat table",
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
        "cols": "Explicit list of status columns",
        "counters": "Explicit list of keys to show totals for in ``py{x} -c``",
        "cons": 'Constraints on run matrix keys, e.g. ``"mach>1.0"``',
        "cutoff": "Min file size or count for 'large'",
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
        "hide-cols": "Standard columns to hide in run matrix status table",
        "hide-counters": "Standard keys to omit totals after run mat table",
        "imax": "Do not extend a case beyond iteration *M*",
        "incremental": "Run case for one phase [or stop after *STOP_PHASE*]",
        "iter-fm": "Extract iterative force & moment histories",
        "j": "List PBS/Slurm job ID in ``-c`` output",
        "kill": "Remove jobs from the queue and stop them",
        "ll": "Extract line load data [comps matching *PAT*] for case(s)",
        "marked": "Show only cases marked either PASS or ERROR",
        "n": "Submit at most *N* cases",
        "pat": "Consider file names matching pattern *PAT*",
        "pt": "Extract surf point sensors [comps matching *PAT*] for case(s)",
        "prompt": "Don't ask for confirmation when deleting cases w/o iters",
        "prop": "Extract scalar properties [comps matching *PAT*]",
        "q": "Submit to PBS/Slurm queue *QUEUE*, overrides value in JSON file",
        "qsub": "Don't submit PBS/Slurm jobs even if otherwise specified",
        "re": "Limit to cases containing regular expression *REGEX*",
        "report": "Generate the report *RP* or the first in the list",
        "restart": "When submitting new jobs, only submit new cases",
        "surcp": "Extract surface pressure data for case(s)",
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
        "add-cols": "COLS",
        "add-counters": "COLS",
        "cons": "CONS",
        "counters": "COLS",
        "cutoff": "SIZE",
        "dbpyfunc": "[PAT]",
        "e": "EXEC",
        "extend": "[N_EXT]",
        "f": "JSON",
        "filter": "TXT",
        "fm": "[PAT]",
        "glob": "PAT",
        "hide-cols": "COLS",
        "hide-counters": "COLS",
        "imax": "M",
        "incremental": "[STOP_PHASE]",
        "iter-fm": "[PAT]",
        "ll": "[PAT]",
        "n": "N",
        "pat": "PAT",
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


# Settings for any caseloop command
class _CfdxCaseLoopArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Additional options
    _optlist = (
        "add-cols",
        "add-counters",
        "cols",
        "counters",
        "hide-cols",
        "hide-counters",
        "j",
    )

    # Common aliases
    _optmap = {
        "add": "add-cols",
    }


# Settings for databook commands
class _CfdxExtractArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape-1to2"

    # Description
    _help_title = "Convert Python modules for upgrade CAPE 1 to 2"

    # Additional options
    _optlist = (
        "delete",
    )


# Settings for CAPE 1to2
class Cfdx1to2Args(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Options
    _optlist = (
        "h",
        "1to2",
    )

    # Defaults
    _rc = {
        "1to2": True,
    }


# Settings for -c
class CfdxCheckArgs(_CfdxCaseLoopArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-check"

    # Description
    _help_title = "Check status of one or more cases"

    # Additional options
    _optlist = (
        "c",
        "u",
    )


# Settings for --apply
class CfdxApplyArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-apply"

    # Description
    _help_title = "Re-apply current settings to case(s)"

    # Additional options
    _optlist = (
        "apply",
        "qsub",
    )

    # Defaults
    _rc = {
        "qsub": False,
    }


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


# Settings for --batch
class CfdxBatchArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-batch"

    # Description
    _help_title = "Submit CAPE command as a PBS/Slurm batch job"


# Settings for --check-db
class CfdxCheckDBArgs(_CfdxSubsetArgs):
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
class CfdxCheckFMArgs(_CfdxSubsetArgs):
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
class CfdxCheckLLArgs(_CfdxSubsetArgs):
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
class CfdxCheckTriqFMArgs(_CfdxSubsetArgs):
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
        "qsub": False,
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


# Settings for --iter-fm
class CfdxExtractIterFMArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-iter-fm"

    # Description
    _help_title = "Extract iterative force & moment histories"

    # Additional options
    _optlist = (
        "iter-fm",
    )

    # Positional parameters
    _arglist = (
        "iter-fm",
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
class CfdxExtractSurfCpArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-extract-surfcp"

    # Description
    _help_title = "Extract surfcp data"

    # Additional options
    _optlist = (
        "surfcp",
    )

    # Positional parameters
    _arglist = (
        "surfcp",
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
class CfdxExtractTriqPTArgs(_CfdxExtractArgs):
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


# Settings for --find-json
class CfdxFindJSONArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-find-json"

    # Description
    _help_title = "Find CAPE JSON files"

    # Arguments
    _arglist = (
        "pat",
    )

    # Additional options
    _optlist = (
        "h",
        "pat",
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


# Settings for --find-large
class CfdxFindLargeArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-find-large"

    # Description
    _help_title = "Find folders with large file size"

    # Additional options
    _optlist = (
        "cutoff",
    )

    # Arguemnts
    _arglist = (
        "cutoff",
    )


# Settings for --qdel
class CfdxQdelArgs(_CfdxCaseLoopArgs):
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
class CfdxRemoveCasesArgs(_CfdxCaseLoopArgs):
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


# Settings for --search-large
class CfdxSearchLargeArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cfdx-search-large"

    # Description
    _help_title = "Find large cases from all run matrices in repo"

    # Additional options
    _optlist = (
        "pat",
        "cutoff",
    )

    # Arguemnts
    _arglist = (
        "pat",
        "cutoff",
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
    _cntl_mod = "cape.cfdx.cntl"
    _casecntl_mod = "cape.cfdx.casecntl"

    # List of available options
    _optlist = (
        "1to2",
        "FAIL",
        "I",
        "PASS",
        "apply",
        "add-cols",
        "add-counters",
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
        "cols",
        "cons",
        "counters",
        "cutoff",
        "dbpyfunc",
        "delete",
        "dezombie",
        "e",
        "extend",
        "f",
        "filter",
        "fm",
        "force",
        "glob",
        "h",
        "hide-cols",
        "hide-counters",
        "imax",
        "incremental",
        "iter-fm",
        "j",
        "kill",
        "ll",
        "marked",
        "n",
        "pat",
        "pt",
        "prompt",
        "q",
        "qdel",
        "qsub",
        "re",
        "report",
        "restart",
        "skeleton",
        "surfcp",
        "rm",
        "start",
        "triqfm",
        "u",
        "unarchive",
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
        "1to2",
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
        "extract-iter-fm",
        "extract-ll",
        "extract-pyfunc",
        "extract-prop",
        "extract-surfcp",
        "extract-timeseries",
        "extract-triqfm",
        "extract-triqpt",
        "fail",
        "find-json",
        "find-large",
        "qdel",
        "report",
        "rm",
        "search-large",
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
        "1to2": Cfdx1to2Args,
        "archive": CfdxArchiveArgs,
        "apply": CfdxApplyArgs,
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
        "extract-iter-fm": CfdxExtractIterFMArgs,
        "extract-ll": CfdxExtractLLArgs,
        "extract-pyfunc": CfdxExtractPyFuncArgs,
        "extract-prop": CfdxExtractPropArgs,
        "extract-surfcp": CfdxExtractSurfCpArgs,
        "extract-timeseries": CfdxExtractTimeSeriesArgs,
        "extract-triqfm": CfdxExtractTriqFMArgs,
        "extract-triqpt": CfdxExtractTriqPTArgs,
        "fail": CfdxFailArgs,
        "find-json": CfdxFindJSONArgs,
        "find-large": CfdxFindLargeArgs,
        "qdel": CfdxQdelArgs,
        "report": CfdxReportArgs,
        "rm": CfdxRemoveCasesArgs,
        "run": CfdxRunArgs,
        "search-large": CfdxSearchLargeArgs,
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
            # Check if present
            if opt in self:
                # Get value
                if self[opt] in (True, False):
                    # Remove that flag
                    self.pop_opt_param(opt)
                # Return the command name
                return cmdname
        # Default is "start"
        return "start"


def cape_1to2(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --1to2`` command

    :Call:
        >>> ierr == cape_1to2(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2025-01-03 ``@ddalle``: v1.0
    """
    print("Updating CAPE 1 -> 2")
    convert1to2.upgrade1to2()
    return 0


def cape_apply(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --apply`` command

    :Call:
        >>> ierr == cape_apply(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2025-01-03 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.ApplyCases(**kw)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.MarkPASS(**kw)
    # Return code
    return IERR_OK


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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.ArchiveCases(**kw)
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
    # Read instance
    cntl, _ = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.ExecScript(**kw)
    # Return code
    return IERR_OK


def cape_extend(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --extend`` command

    :Call:
        >>> ierr == cape_extend(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.UpdateFM(**kw)
    # Return code
    return IERR_OK


def cape_extract_iterfm(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --iter-fm`` command

    :Call:
        >>> ierr == cape_extract_iterfm(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.UpdateIterFM(**kw)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.UpdatePyFuncDataBook(**kw)
    # Return code
    return IERR_OK


def cape_extract_surfcp(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --surfcp`` command

    :Call:
        >>> ierr == cape_extract_surfcp(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-28 ``@ddalle``: v1.0
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.UpdateSurfCp(**kw)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.MarkERROR(**kw)
    # Return code
    return IERR_OK


def cape_find_json(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --find-json`` command

    :Call:
        >>> ierr == cape_find_json(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2025-09-25 ``@ddalle``: v1.0
    """
    # Parse args
    kw = parser.get_kwargs()
    # Find files
    json_files = manage.find_json(kw.get("pat"))
    # List them
    for fname in json_files:
        print(fname)
    # Return code
    return IERR_OK


def cape_find_large(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --find-large`` command

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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.find_large_cases(**kw)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.qdel_cases(**kw)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run command
    cntl.UpdateReport(**kw)
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
        * 2025-06-20 ``@ddalle``: v1.1; use `rm_cases()`
    """
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.rm_cases(**kw)
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
    # Read instance
    runner, _ = read_runner_kwargs(parser)
    # Run the case
    runner.run()
    # Return code
    return IERR_OK


def cape_search_large(parser: CfdxArgReader) -> int:
    r"""Run the ``cape --search-large`` command

    :Call:
        >>> ierr == cape_search_large(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            Parsed CLI args
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2025-09-25 ``@ddalle``: v1.0
    """
    # Parse args
    kw = parser.get_kwargs()
    # Find large cases
    manage.search_repo_large(**kw)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
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
    # Read instance
    cntl, kw = read_cntl_kwargs(parser)
    # Run the command
    cntl.UnmarkCase(**kw)
    # Return code
    return IERR_OK


def read_cntl_kwargs(parser: CfdxArgReader):
    r"""Read a CAPE run matrix control instance of appropriate class

    :Call:
        >>> cntl, kw = read_cntl_kwargs(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            CLI parser instance
    :Outputs:
        *cntl*: :class:`cape.cfdx.cntl.Cntl`
            CAPE run matrix control instance (solver-specific)
        *kw*: :class:`dict`
            Preprocessed keyword arguments
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
        * 2025-01-24 ``@ddalle``: v2.0; use module name instead of cls
    """
    # Get file name
    fname = parser.get_opt("f")
    # Get module name
    modname = parser.cntl_mod
    # Import it
    cntlmod = importlib.import_module(modname)
    # Instantiate
    cntl = cntlmod.Cntl(fname)
    # Parse arguments
    kw = parser.get_kwargs()
    # Preprocess
    cntl.preprocess_kwargs(kw)
    # Log
    cntl.log_parser(parser)
    # Output
    return cntl, kw


def read_runner_kwargs(parser: CfdxArgReader):
    r"""Read a CAPE case runner instance to interact with CFD in ``PWD``

    :Call:
        >>> runner = read_runner(parser)
    :Inputs:
        *parser*: :class:`CfdxArgReader`
            CLI parser instance
    :Outputs:
        *runner*: :class:`cape.cfdx.casecntl.CaseRunner`
            Case runner instance to control case in current folder
        *kw*: :class:`dict`
            Preprocessed keyword arguments
    :Versions:
        * 2024-12-30 ``@ddalle``: v1.0
        * 2025-01-24 ``@ddalle``: v2.0; use module name instead of cls
    """
    # Import module
    mod = importlib.import_module(parser.casecntl_mod)
    # Instantiate
    runner = mod.CaseRunner()
    # Parse arguments
    kw = parser.get_kwargs()
    # Output
    return runner, kw


# Name -> Function
CMD_DICT = {
    "1to2": cape_1to2,
    "apply": cape_apply,
    "approve": cape_approve,
    "archive": cape_archive,
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
    "extract-iter-fm": cape_extract_iterfm,
    "extract-ll": cape_extract_ll,
    "extract-prop": cape_extract_prop,
    "extract-pyfunc": cape_extract_pyfunc,
    "extract-surfcp": cape_extract_surfcp,
    "extract-timeseries": cape_extract_timeseries,
    "extract-triqfm": cape_extract_triqfm,
    "extract-triqpt": cape_extract_triqpt,
    "fail": cape_fail,
    "find-json": cape_find_json,
    "find-large": cape_find_large,
    "qdel": cape_qdel,
    "report": cape_report,
    "rm": cape_rm,
    "run": cape_run,
    "search-large": cape_search_large,
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
    cmdname, subparser, ierr = parser.fullparse_check(argv)
    # Check for errors
    if ierr:
        return IERR_OPT
    # Check for valid command name or other front-desk help triggers
    if parser.help_frontdesk(cmdname):
        return IERR_OK
    # Check for ``-h``
    if subparser.show_help("h"):
        return IERR_OK
    # Set Cntl/CaseRunner classes for this solver
    subparser.cntl_mod = parser_cls._cntl_mod
    subparser.casecntl_mod = parser_cls._casecntl_mod
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
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``cape-cfdx``

    This is basically an interface to :func:`cape.cfdx.cntl.Cntl.cli`.

    :Call:
        >>> main()
    :Versions:
        * 2021-03-04 ``@ddalle``: v1.0
    """
    return main1(argv)


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
