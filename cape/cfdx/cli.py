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
from typing import Any, Optional, Union, Tuple

# CAPE modules
from . import manage
from .. import capeconfig
from .. import sysutils
from ..argread import ArgReader, ArgReadError, BOOL_TYPES, INT_TYPES
from ..errors import CapeError, CapeValueError


# Constants
IERR_OK = 0
IERR_INTERRUPT = 2
IERR_CMD = 16
IERR_OPT = 32
IERR_RUNTIME = 128

# Inferred commands from options
CMD_NAMES = {
    "1to2": "1to2",
    "batch": "batch",
    "c": "check",
    "PASS": "approve",
    "FAIL": "fail",
    "unmark": "unmark",
    "defail": "defail",
    "dezombie": "dezombie",
    "edit": "edit-json",
    "extend": "extend",
    "apply": "apply",
    "dbpyfunc": "extract-pyfunc",
    "dex": "extract",
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
    "ui": "ui",
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
class CfdxArgReader(ArgReader):
    # No attributes
    __slots__ = (
        "cntl_mod",
        "casecntl_mod",
        "cntl_cls",
        "runner_cls",
    )

    # Common options
    _optlist = (
        "h",
        "solver",
    )

    # Common aliases
    _optmap = {
        "ERROR": "FAIL",
        "F": "force",
        "add-col": "add-cols",
        "add_col": "add-cols",
        "add_cols": "add-cols",
        "aero": "fm",
        "approve": "PASS",
        "check": "c",
        "checkDB": "check-db",
        "checkFM": "check-fm",
        "checkLL": "check-ll",
        "checkTriqFM": "check-triqfm",
        "const": "constant",
        "constraints": "cons",
        "early-exit": "early",
        "edit-json": "edit",
        "exec": "e",
        "fail": "FAIL",
        "file": "f",
        "help": "h",
        "hide": "hide-cols",
        "iterfm": "iter-fm",
        "json": "f",
        "kill": "qdel",
        "minsize": "cutoff",
        "nbatch": "batchsize",
        "output-json": "o",
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
        "adaptive": bool,
        "add-cols": (str, list),
        "add-counters": (str, list),
        "apply": bool,
        "archive": bool,
        "auto": bool,
        "batch": bool,
        "batchsize": int,
        "blend": bool,
        "c": bool,
        "check-db": bool,
        "check-fm": bool,
        "check-ll": bool,
        "check-triqfm": bool,
        "clean": bool,
        "cols": (str, list),
        "compile": bool,
        "cons": str,
        "constant": bool,
        "counters": (str, list),
        "cmd": str,
        "cutoff": str,
        "dbpyfunc": (bool, str),
        "delete": bool,
        "dex": (bool, str),
        "defail": bool,
        "dezombie": bool,
        "e": str,
        "early": bool,
        "edit": str,
        "extend": (bool, int),
        "f": str,
        "failed": bool,
        "filter": str,
        "fixed": bool,
        "fjson": str,
        "force": bool,
        "fpdf": str,
        "fm": (bool, str),
        "glob": str,
        "h": bool,
        "hide-cols": (str, list),
        "hide-counters": (str, list),
        "imax": int,
        "incremental": bool,
        "iter-fm": (bool, str),
        "j": bool,
        "local": bool,
        "ll": (bool, str),
        "marked": bool,
        "me": bool,
        "n": int,
        "nmax": int,
        "nproc": int,
        "nsurf": int,
        "o": str,
        "opt": str,
        "passed": bool,
        "pat": str,
        "pats": str,
        "prompt": bool,
        "prop": (bool, str),
        "pt": (bool, str),
        "pull": bool,
        "q": bool,
        "qdel": bool,
        "qsub": bool,
        "raw": bool,
        "re": str,
        "remote-dir": str,
        "report": (bool, str),
        "restart": bool,
        "rm": bool,
        "skeleton": bool,
        "solver": str,
        "start": bool,
        "status": str,
        "surfcp": (bool, str),
        "triqfm": (bool, str),
        "ts": (bool, str),
        "u": str,
        "ui": bool,
        "unarchive": bool,
        "unmark": bool,
        "unmarked": bool,
        "user": str,
        "v": bool,
        "val": str,
        "wait": bool,
        "x": str,
    }

    # Allowed types prior to conversion
    _rawopttypes = {
        "extend": BOOL_TYPES + INT_TYPES + (str,),
    }

    # Conversion functions
    _optconverters = {
        "batchsize": int,
        "extend": _true_int,
        "imax": int,
        "n": int,
        "nmax": int,
        "nproc": int,
        "nsurf": int,
    }

    # List of options that cannot take a "value"
    _optlist_noval = (
        "1to2",
        "PASS",
        "FAIL",
        "apply",
        "auto",
        "batch",
        "blend",
        "c",
        "compile",
        "early",
        "defail",
        "delete",
        "dezombie",
        "force",
        "incremental",
        "j",
        "local",
        "marked",
        "prompt",
        "pull",
        "qsub",
        "qdel",
        "restart",
        "rm",
        "start",
        "ui",
        "unarchive",
        "unmark",
        "unmarked",
        "v",
        "wait",
    )

    # List of options that usually take a file name value
    _optlist_file = (
        "f",
        "fjson",
        "fpdf",
        "x",
    )

    # Translations for option values
    _optvalmap = {
        "solver": {
            "cart": "pycart",
            "cart3d": "pycart",
            "f3d": "pyfun",
            "fun3d": "pyfun",
            "kestrel": "pykes",
            "lava": "pylava",
            "ofl": "pyover",
            "overflow": "pyover",
        },
    }

    # Allowed values
    _optvals = {
        "solver": (
            "pycart",
            "pyfun",
            "pykes",
            "pylava",
            "pylch",
            "pyover",
        ),
        "status": (
            "---",
            "INCOMP",
            "QUEUE",
            "ERROR",
            "FAIL",
            "DONE",
            "PASS",
            "PASS*",
            "ZOMBIE",
        ),
    }

    # Description of each option
    _help_opt = {
        "1to2": "Convert Python modules in current folder from CAPE 1 to 2",
        "FAIL": "Mark case(s) as ERRORs",
        "I": "Specific case indices, e.g. ``-I 4:8,12``",
        "PASS": "Mark case(s) as PASS",
        "adaptive": "Save the adapted-mesh version of flow data (more data)",
        "add-cols": "Additional columns to show in run matrix status table",
        "add-counters": "Additional keys to show totals after run mat table",
        "apply": "Apply current JSON settings to existing case(s)",
        "archive": "Archive files from case(s) and delete extra files",
        "auto": "Ignore *RunControl* > *NJob* if set",
        "batch": "Submit PBS/Slurm job and run this command there",
        "batchsize": "Number of snapshots to collect into each batch files",
        "blend": "Combine option values rather than overwrite",
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
        "constant": "Assume a fixed (not adaptive) mesh for data collection",
        "cutoff": "Min file size or count for 'large'",
        "dbpyfunc": "Extract scalar data from custom Python function",
        "delete": "Delete DataBook entries instead of adding new ones",
        "dex": "Extract DataBook components matching pattern *DEX*",
        "defail": "Clean up FAIL cases, deletes ``FAIL`` and others",
        "dezombie": "Clean up ZOMBIE cases, RUNNING but no recent file mods",
        "e": "Execute the command *EXEC*",
        "early": "Reduce *PhaseIters* to current iter; makes case ``DONE``",
        "edit": "Text of JSON settings to edit and rewrite",
        "extend": "Extend case(s) by *N_EXT* copies of last phase",
        "f": "Use the JSON (or YAML) file *JSON*",
        "filter": "Limit to cases containing the string *TXT*",
        "fixed": "Interpolate flow data to common grid",
        "fjson": "Apply settings from file *FJSON*",
        "fpdf": "Name of PDF file to open/write/send",
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
        "local": "Force an action to be done locally, without SSH transfer",
        "ll": "Extract line load data [comps matching *PAT*] for case(s)",
        "marked": "Show only cases marked either PASS or ERROR",
        "me": "Limit to cases owned by current user (equiv. ``--user $USER``)",
        "n": "Submit at most *N* cases",
        "nmax": "Maximum number of snapshots to process",
        "nproc": "Number of parallel processes to use",
        "nsurf": "Index of surface to process",
        "o": "Name of output JSON file (defaults to same as *f*)",
        "opt": "CAPE user configuration variable name",
        "pat": "Consider file names matching pattern *PAT*",
        "pats": "Additional file name patterns",
        "pt": "Extract surf point sensors [comps matching *PAT*] for case(s)",
        "pull": "Pull remote file to local dir before opening",
        "prompt": "Don't ask for confirmation when deleting cases w/o iters",
        "prop": "Extract scalar properties [comps matching *PAT*]",
        "q": "Submit to PBS/Slurm queue *QUEUE*, overrides value in JSON file",
        "qdel": "Delete a PBS/Slurm job",
        "qsub": "Don't submit PBS/Slurm jobs even if otherwise specified",
        "re": "Limit to cases containing regular expression *REGEX*",
        "remote-dir": "Base path to use on remote host",
        "report": "Generate the report *RP* or the first in the list",
        "restart": "When submitting new jobs, only submit new cases",
        "skeleton": "Delete most files from indicaded PASSED cases",
        "status": "Find cases with a specific status",
        "solver": "Name of CAPE module to use (or determine automatically)",
        "surf": "Name of surface to collect/process",
        "surfcp": "Extract surface pressure data for case(s)",
        "raw": "Collect raw flow data w/o triangulating or interpolating",
        "rm": "Remove indicated cases",
        "start": "Set up but do not start (or submit) cases",
        "triqfm": "Extract triq F&M data [comps matching *PAT*] for case(s)",
        "ts": "Extract time-series data [comps matching *PAT*]",
        "u": "Pretend to be user *UID*",
        "ui": "Run interactive CAPE user interface",
        "unarchive": "Unarchive one or more cases",
        "unmark": "Remove PASS/ERROR marking for case(s)",
        "unmarked": "Show cases with no PASS/ERROR markings",
        "user": "Restrict to cases with this username",
        "v": "Show additional output while running command",
        "val": "Value to set option to",
        "wait": "Wait until application closed before returning control",
        "x": "Execute Python script *PYSCRIPT* after reading JSON",
    }

    # Name for value of select options in option descriptions
    _help_optarg = {
        "I": "INDS",
        "add-cols": "COLS",
        "add-counters": "COLS",
        "batchsize": "N",
        "cons": "CONS",
        "counters": "COLS",
        "cutoff": "SIZE",
        "dbpyfunc": "[PAT]",
        "dex": "[DEX]",
        "e": "EXEC",
        "edit": "SETTINGS",
        "extend": "[N_EXT]",
        "f": "JSON",
        "filter": "TXT",
        "fjson": "FJSON",
        "fm": "[PAT]",
        "fpdf": "PDFFILE",
        "glob": "PAT",
        "hide-cols": "COLS",
        "hide-counters": "COLS",
        "imax": "M",
        "incremental": "[STOP_PHASE]",
        "iter-fm": "[PAT]",
        "ll": "[PAT]",
        "n": "N",
        "nmax": "NMAX",
        "nproc": "NPROC",
        "nsurf": "SURF",
        "o": "OUT_JSON",
        "opt": "OPT",
        "pat": "PAT",
        "pats": "PATTERNS",
        "prop": "[PAT]",
        "pt": "[PAT]",
        "q": "QUEUE",
        "re": "REGEX",
        "remote-dir": "DIRNAME",
        "report": "[RP]",
        "solver": "SOLVER",
        "surf": "SURF",
        "status": "STATUS",
        "triqfm": "[PAT]",
        "ts": "[PAT]",
        "u": "UID",
        "user": "USER",
        "val": "VALUE",
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
        "me",
        "re",
        "status",
        "unmarked",
        "user",
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
        "solver",
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

    # Name of function
    _name = "cape 1to2"

    # Description
    _help_title = "Convert Python and JSON files from CAPE 1 to 2 standard"

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
    _name = "cape check"

    # Description
    _help_title = "Check status of one or more cases"

    # Additional options
    _optlist = (
        "c",
        "nproc",
        "u",
    )

    # Default values
    _rc = {
        "nproc": 8,
    }


# Settings for --apply
class CfdxApplyArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape apply"

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
    _name = "cape approve"

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
    _name = "cape archive"

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
    _name = "cape batch"

    # Description
    _help_title = "Submit CAPE command as a PBS/Slurm batch job"


# Settings for --check-db
class CfdxCheckDBArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape check-db"

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
    _name = "cape check-fm"

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
    _name = "cape check-ll"

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
    _name = "cape check-triqfm"

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
    _name = "cape clean"

    # Description
    _help_title = "Remove extra files not necessary for running a case"

    # Additional options
    _optlist = (
        "clean",
    )


# Settings for collect-surf
class CfdxCollectSurfArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape collect-surf"

    # Description
    _help_title = "Collect surface data in current case folder"

    # Options
    _optlist = (
        "h",
        "nsurf",
        "batchsize",
        "clean",
        "nmax",
        "nproc",
    )

    # Aliases
    _optmap = {
        "surf": "nsurf",
    }

    # Positional parameters
    _arglist = (
        "nsurf",
    )

    # Defaults
    _rc = {
        "clean": False,
        "nsurf": 1,
    }


# Settings for collect-cutplane
class CfdxCollectCutPlaneArgs(CfdxCollectSurfArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape collect-cutplane"

    # Additional options
    _optlist = (
        "adaptive",
        "constant",
        "fixed",
        "nproc",
        "raw",
    )

    # Description
    _help_title = "Collect cut-plane data in current case folder"

    # Defaults
    _rc = {
        "nsurf": None,
    }


# Settings for --defail
class CfdxDefailArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape defail"

    # Description
    _help_title = "Remove FAIL files from case(s)and other failure artifacts"

    # Additional options
    _optlist = (
        "defail",
    )


# Settings for --dezombie
class CfdxDezombieArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape dezombie"

    # Description
    _help_title = "Delete job and clean-up stalled cases (aka 'zombie' cases)"

    # Additional options
    _optlist = (
        "dezombie",
        "early",
    )

    # Aliases
    _optmap = {
        "early-exit": "early",
    }


# Settings for --edit
class CfdxEditArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape edit-json"

    # Description
    _help_title = "Edit JSON settings from command-line"

    # Options
    _optlist = (
        "h",
        "f",
        "edit",
        "fjson",
    )

    # Alternate aliases
    _optmap = {
        "json": "fjson",
    }

    # Positional parameters
    _arglist = (
        "edit",
    )

    # Required options
    _optlistreq = (
        "edit",
    )


# Settings for -e
class CfdxExecArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape exec"

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
    _name = "cape extend"

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


# Settings for --dex
class CfdxExtractDexArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape extract"

    # Description
    _help_title = "Extract DataBook components"

    # Additional options
    _optlist = (
        "dex",
    )

    # Positional parameters
    _arglist = (
        "dex",
    )


# Settings for --fm
class CfdxExtractFMArgs(_CfdxExtractArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape extract-fm"

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
    _name = "cape extract-iter-fm"

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
    _name = "cape extract-ll"

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
    _name = "cape extract-prop"

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
    _name = "cape extract-pyfunc"

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
    _name = "cape extract-surfcp"

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
    _name = "cape extract-timeseries"

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
    _name = "cape extract-triqfm"

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
    _name = "cape extract-triqpt"

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
    _name = "cape find-json"

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
    _name = "cape fail"

    # Description
    _help_title = "Mark selected cases as ERRORs"

    # Additional options
    _optlist = (
        "FAIL",
    )


# Settings for --find
class CfdxFindArgs(_CfdxCaseLoopArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape find"

    # Description
    _help_title = "Find indices of cases meeting constraints"


# Settings for --find-large
class CfdxFindLargeArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape find-large"

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


# Settings for get-config
class CfdxGetConfigArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape get-config"

    # Description
    _help_title = "Show value of user CAPE configuration"

    # Additional options
    _optlist = (
        "opt",
    )

    # Required options
    _optlistreq = (
        "opt",
    )

    # Arguemnts
    _arglist = (
        "opt",
    )


# Settings for open-pdf
class CfdxOpenPDFArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape open-pdf"

    # Description
    _help_title = "Open a PDF file in preferred reader"

    # Additional options
    _optlist = (
        "fpdf",
        "local",
        "pull",
        "remote-dir",
        "wait",
    )

    # Required options
    _optlistreq = (
        "fpdf",
    )

    # Arguemnts
    _arglist = (
        "fpdf",
    )


# Settings for set-config
class CfdxPostFileArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape post-file"

    # Minimum args
    _nargmin = 1

    # Arg names
    _optlist = (
        "pat",
        "v",
    )
    _arglist = (
        "pat",
        "pats",
    )

    # Description
    _help_title = "Send one or more files to *RemoteHost*"


# Settings for --qdel
class CfdxQdelArgs(_CfdxCaseLoopArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape qdel"

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


# Settings for receive-file
class CfdxReceiveFileArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape receive-file"

    # Description
    _help_title = "Receive posted file from remote host"


# Settings for --report
class CfdxReportArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape report"

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
    _name = "cape rm"

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
    _name = "cape run"

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
    _name = "cape search-large"

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


# Settings for set-config
class CfdxSetConfigArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape set-config"

    # Description
    _help_title = "Set value of a user CAPE configuration"

    # Additional options
    _optlist = (
        "opt",
        "val",
        "blend",
    )

    # Required options
    _optlistreq = (
        "opt",
        "val",
    )

    # Arguemnts
    _arglist = (
        "opt",
        "val",
    )


# Settings for --skeleton
class CfdxSkeletonArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape skeleton"

    # Description
    _help_title = "Clean up case folder; leave only key files"

    # Additional options
    _optlist = (
        "skeleton",
    )


# Settings for interactive UI
class CfdxUIArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape ui"

    # Description
    _help_title = "Run interactive CAPE user interface"

    # Options
    _optlist = (
        "ui",
    )


# Settings for -n
class CfdxStartArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape start"

    # Description
    _help_title = "Setup, start, and/or submit cases"

    # Additional options
    _optlist = (
        "auto",
        "n",
        "j",
        "q",
        "qsub",
        "u",
        "start",
    )

    # Default values
    _rc = {
        "restart": True,
        "start": True,
    }


# Settings for triangulate-cutplane
class CfdxTriangulateCutPlaneArgs(CfdxArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape triangulate-cutplane"

    # Description
    _help_title = "Triangulate oversetting cut-plane flow viz files"

    # Options
    _optlist = (
        "h",
        "nproc",
        "nsurf",
        "clean",
        "nmax",
    )

    # Positional paramters
    _arglist = (
        "nsurf",
    )

    # Aliases
    _optmap = {
        "surf": "nsurf",
    }

    # Defaults
    _rc = {
        "clean": False,
    }


# Settings for --unarchive
class CfdxUnarchiveArgs(_CfdxSubsetArgs):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "cape unarchive"

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
    _name = "cape unmark"

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
        "adaptive",
        "apply",
        "add-cols",
        "add-counters",
        "archive",
        "auto",
        "batch",
        "batchsize",
        "c",
        "check-db",
        "check-fm",
        "check-ll",
        "check-triqfm",
        "clean",
        "compile",
        "cols",
        "cons",
        "constant",
        "counters",
        "cutoff",
        "dbpyfunc",
        "defail",
        "delete",
        "dezombie",
        "dex",
        "e",
        "edit",
        "extend",
        "f",
        "filter",
        "fixed",
        "fjson",
        "fm",
        "force",
        "fpdf",
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
        "local",
        "marked",
        "me",
        "n",
        "nmax",
        "nproc",
        "nsurf",
        "o",
        "opt",
        "pat",
        "pt",
        "pull",
        "prompt",
        "q",
        "qdel",
        "qsub",
        "raw",
        "re",
        "remote-dir",
        "report",
        "restart",
        "skeleton",
        "surf",
        "surfcp",
        "rm",
        "start",
        "status",
        "triqfm",
        "u",
        "unarchive",
        "unmark",
        "unmarked",
        "user",
        "val",
        "wait",
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
        "collect-cutplane",
        "collect-surf",
        "defail",
        "dezombie",
        "edit-json",
        "exec",
        "extend",
        "extract",
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
        "find",
        "find-cases",
        "find-json",
        "find-large",
        "get-config",
        "open-pdf",
        "post-file",
        "qdel",
        "receive-file",
        "report",
        "rm",
        "search-large",
        "set-config",
        "skeleton",
        "triangulate-cutplane",
        "ui",
        "unarchive",
        "unmark",
    )

    # Alternate command names
    _cmdmap = {
        "c": "check",
        "collect-cut": "collect-cutplane",
        "collect-cutp": "collect-cutplane",
        "collect-plane": "collect-cutplane",
        "collect-surfdata": "collect-surf",
        "config": "set-config",
        "dex": "extract",
        "e": "exec",
        "edit": "edit-json",
        "error": "fail",
        "find": "find-cases",
        "get": "get-config",
        "get-file": "receive-file",
        "mark-error": "fail",
        "mark-failure": "fail",
        "mark-pass": "approve",
        "pass": "approve",
        "post-files": "post-file",
        "qsub": "start",
        "r": "run",
        "receive-files": "receive-file",
        "send-file": "post-file",
        "send-files": "post-file",
        "set": "set-config",
        "submit": "start",
        "triplane": "triangulate-cutplane",
        "tri-plane": "triangulate-cutplane",
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
        "collect-cutplane": CfdxCollectCutPlaneArgs,
        "collect-surf": CfdxCollectSurfArgs,
        "defail": CfdxDefailArgs,
        "dezombie": CfdxDezombieArgs,
        "edit-json": CfdxEditArgs,
        "exec": CfdxExecArgs,
        "extend": CfdxExtendArgs,
        "extract": CfdxExtractDexArgs,
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
        "find": CfdxFindArgs,
        "find-json": CfdxFindJSONArgs,
        "find-large": CfdxFindLargeArgs,
        "get-config": CfdxGetConfigArgs,
        "open-pdf": CfdxOpenPDFArgs,
        "post-file": CfdxPostFileArgs,
        "qdel": CfdxQdelArgs,
        "receive-file": CfdxReceiveFileArgs,
        "report": CfdxReportArgs,
        "rm": CfdxRemoveCasesArgs,
        "run": CfdxRunArgs,
        "search-large": CfdxSearchLargeArgs,
        "set-config": CfdxSetConfigArgs,
        "start": CfdxStartArgs,
        "skeleton": CfdxSkeletonArgs,
        "triangulate-cutplane": CfdxTriangulateCutPlaneArgs,
        "ui": CfdxUIArgs,
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
        "n",
        "I",
        "cons",
        "re",
        "me",
        "marked",
        "unmarked",
        "batch",
        "e",
        "restart",
        "start",
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
        # Default is "start" unless no optiosn given
        if len(self.keys()) == 0:
            return "ui"
        else:
            return "start"


@Cfdx1to2Args.rst
def cape_1to2(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    print("Updating CAPE 1 -> 2")
    from cape import convert1to2
    convert1to2.upgrade1to2()
    return IERR_OK, None


@CfdxApplyArgs.rst
def cape_apply(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxApplyArgs, *a, **kw)
    # Run the command
    v = cntl.ApplyCases(**kw)
    # Output
    return IERR_OK, v


@CfdxApproveArgs.rst
def cape_approve(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # REad *cntl*
    cntl, kw = read_cntl(CfdxApproveArgs, *a, **kw)
    # Run command
    v = cntl.MarkPASS(**kw)
    # Output
    return IERR_OK, v


@CfdxArchiveArgs.rst
def cape_archive(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # REad *cntl*
    cntl, kw = read_cntl(CfdxArchiveArgs, *a, **kw)
    # Run command
    v = cntl.ArchiveCases(**kw)
    # Output
    return IERR_OK, v


@CfdxBatchArgs.rst
def cape_batch(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read instance
    cntl, _ = read_cntl(CfdxBatchArgs, *a, **kw)
    # Get a parser
    parser = CfdxBatchArgs()
    # Save the kwargs
    parser.kwargs_double_dash = kw
    # Select the program based on the resulting module
    parser.prog = cntl.__module__.split('.')[1]
    # Reconstruct command-line args
    argv = parser.reconstruct()
    # Remove ``-batch`` from command name
    cmdname = argv[0]
    if cmdname.endswith("-batch"):
        argv[0] = cmdname.rsplit('-', 1)[0]
    # Check for explicit executable
    pyexec = cntl.opts.get_PythonExec()
    if pyexec:
        # Get name of module, e.g "cape.pyfun"
        modname = cntl.__module__.rsplit('.', 1)[0]
        # Full name of module: "cfdx" -> "cape.cfdx"
        argv[0] = modname
        # Prepend python3 -m ...
        argv = [pyexec, '-m'] + argv
    # Remove recursive batch
    if "--batch" in argv:
        argv.remove("--batch")
    # Run the command
    v = cntl.run_batch(argv)
    # Return code
    return IERR_OK, v


@CfdxCheckArgs.rst
def cape_c(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxCheckArgs, *a, **kw)
    # Run the main command
    v = cntl.DisplayStatus(**kw)
    # Output
    return IERR_OK, v


@CfdxCheckDBArgs.rst
def cape_check_db(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxCheckDBArgs, *a, **kw)
    # Run the command
    cntl.CheckFM(**kw)
    cntl.CheckLL(**kw)
    cntl.CheckTriqFM(**kw)
    # Return code
    return IERR_OK, None


@CfdxCheckFMArgs.rst
def cape_check_fm(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxCheckFMArgs, *a, **kw)
    # Run the command
    v = cntl.CheckFM(**kw)
    # Return code
    return IERR_OK, v


@CfdxCheckLLArgs.rst
def cape_check_ll(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxCheckLLArgs, *a, **kw)
    # Run the command
    v = cntl.CheckLL(**kw)
    # Return code
    return IERR_OK, v


@CfdxCheckTriqFMArgs.rst
def cape_check_triqfm(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxCheckTriqFMArgs, *a, **kw)
    # Run the command
    v = cntl.CheckTriqFM(**kw)
    # Return code
    return IERR_OK, v


@CfdxCleanArgs.rst
def cape_clean(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxCleanArgs, *a, **kw)
    # Run the command
    v = cntl.CleanCases(**kw)
    # Return code
    return IERR_OK, v


@CfdxDefailArgs.rst
def cape_defail(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxDezombieArgs, *a, **kw)
    # Run the command
    v = cntl.Defail(**kw)
    # Return code
    return IERR_OK, v


@CfdxDezombieArgs.rst
def cape_dezombie(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxDezombieArgs, *a, **kw)
    # Run the command
    v = cntl.Dezombie(**kw)
    # Return code
    return IERR_OK, v


@CfdxEditArgs.rst
def cape_edit(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxEditArgs, *a, **kw)
    # Construct inputs
    txt = a[0]
    fjson = kw.get("fjson")
    # Run the command
    v = cntl.edit_json(txt, fjson=fjson)
    # Return code
    return IERR_OK, v


@CfdxExecArgs.rst
def cape_exec(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExecArgs, *a, **kw)
    # Run the command
    v = cntl.ExecScript(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtendArgs.rst
def cape_extend(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtendArgs, *a, **kw)
    # Run the command
    v = cntl.ExtendCases(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractDexArgs.rst
def cape_extract_dex(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractDexArgs, *a, **kw)
    # Run the command
    v = cntl.update_dex(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractFMArgs.rst
def cape_extract_fm(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractFMArgs, *a, **kw)
    # Run the command
    v = cntl.UpdateFM(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractIterFMArgs.rst
def cape_extract_iterfm(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractIterFMArgs, *a, **kw)
    # Run the command
    v = cntl.UpdateIterFM(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractLLArgs.rst
def cape_extract_ll(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractLLArgs, *a, **kw)
    # Run the command
    v = cntl.UpdateLL(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractPropArgs.rst
def cape_extract_prop(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractPropArgs, *a, **kw)
    # Run the command
    v = cntl.UpdateCaseProp(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractPyFuncArgs.rst
def cape_extract_pyfunc(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractPyFuncArgs, *a, **kw)
    # Run the command
    v = cntl.UpdatePyFuncDataBook(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractSurfCpArgs.rst
def cape_extract_surfcp(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractSurfCpArgs, *a, **kw)
    # Run the command
    v = cntl.UpdateSurfCp(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractTimeSeriesArgs.rst
def cape_extract_timeseries(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractTimeSeriesArgs, *a, **kw)
    # Run the command
    v = cntl.UpdateTS(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractTriqFMArgs.rst
def cape_extract_triqfm(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractTriqFMArgs, *a, **kw)
    # Run the command
    v = cntl.UpdateTriqFM(**kw)
    # Return code
    return IERR_OK, v


@CfdxExtractTriqPTArgs.rst
def cape_extract_triqpt(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxExtractTriqPTArgs, *a, **kw)
    # Run the command
    v = cntl.UpdateTriqPoint(**kw)
    # Return code
    return IERR_OK, v


@CfdxFailArgs.rst
def cape_fail(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxFailArgs, *a, **kw)
    # Run the command
    v = cntl.MarkERROR(**kw)
    # Return code
    return IERR_OK, v


@CfdxFindArgs.rst
def cape_find(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Localized inputs
    import contextlib
    from ..util import pyrangestr
    # Suppress STDOUT during GetIndices()
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            # Read *cntl*
            cntl, kw = read_cntl(CfdxFindArgs, *a, **kw)
    # Find cases
    v = cntl.GetIndices(**kw)
    # Display as a string
    print(pyrangestr(v))
    # Output
    return IERR_OK, v


@CfdxFindJSONArgs.rst
def cape_find_json(*a, **kw) -> Tuple[int, list]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Find files
    json_files = manage.find_json_solver(kw.get("pat"))
    # List them
    for fname in json_files:
        print(fname)
    # Return code
    return IERR_OK, json_files


@CfdxFindLargeArgs.rst
def cape_find_large(*a, **kw) -> Tuple[int, list]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read instance
    cntl, kw = read_cntl(CfdxFindLargeArgs, *a, **kw)
    # Run the command
    v = cntl.find_large_cases(**kw)
    # Return code
    return IERR_OK, v


@CfdxGetConfigArgs.rst
def cape_get_config(*a, **kw) -> Tuple[int, list]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Get value
    v = capeconfig.show_cape_config(a[0])
    # Show it
    print(v)
    # Return code
    return IERR_OK, v


@CfdxOpenPDFArgs.rst
def cape_open_pdf(*a, **kw) -> Tuple[int, list]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Get required argument
    fpdf = a[0]
    # Get options
    rdir = kw.get("remote-dir")
    wait = kw.get("wait", False)
    local = kw.get("local", False)
    pull = kw.get("pull", False)
    # Open the pdf
    sysutils.open_pdf(fpdf, remote=rdir, wait=wait, local=local, pull=pull)
    # Return code
    return IERR_OK, None


@CfdxPostFileArgs.rst
def cape_post_file(*a, **kw) -> Tuple[int, list]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Verbose option
    verbose = kw.get("v", False)
    # Get value
    returncode, filenames = sysutils.post_file(*a, v=verbose)
    # Return code
    return returncode, filenames


@CfdxQdelArgs.rst
def cape_qdel(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxQdelArgs, *a, **kw)
    # Run the command
    v = cntl.qdel_cases(**kw)
    # Return code
    return IERR_OK, v


@CfdxReceiveFileArgs.rst
def cape_receive_file(*a, **kw) -> Tuple[int, list]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Get value
    filenames = sysutils.receive_file()
    # Return code
    return IERR_OK, filenames


@CfdxReportArgs.rst
def cape_report(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxReportArgs, *a, **kw)
    # Run command
    v = cntl.UpdateReport(**kw)
    # Return code
    return IERR_OK, v


@CfdxRemoveCasesArgs.rst
def cape_rm(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxRemoveCasesArgs, *a, **kw)
    # Run the command
    v = cntl.rm_cases(**kw)
    # Return code
    return IERR_OK, v


@CfdxRunArgs.rst
def cape_run(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read instance
    runner, kw = read_runner(**kw)
    # Run the case
    v = runner.run()
    # Return code
    return IERR_OK, v


@CfdxCollectSurfArgs.rst
def cape_collect_surfdata(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read instance
    runner, kw = read_runner(**kw)
    # Process args
    nsurf = kw.get("nsurf")
    nbatch = kw.get("batchsize")
    clean = kw.get("clean")
    nmax = kw.get("nmax")
    nproc = kw.get("nproc")
    # Run the case
    v = runner.collect_surfdata(
        nsurf=nsurf,
        nbatch=nbatch,
        clean=clean,
        nmax=nmax,
        nproc=nproc)
    # Return code
    return IERR_OK, v


@CfdxCollectCutPlaneArgs.rst
def cape_collect_cutplane(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read instance
    runner, kw = read_runner(**kw)
    # Process args
    nsurf = kw.get("nsurf")
    nbatch = kw.get("batchsize")
    clean = kw.get("clean")
    nmax = kw.get("nmax")
    nproc = kw.get("nproc")
    # Process mode options
    modes = [
        opt for opt in ("adaptive", "constant", "fixed", "raw")
        if kw.get(opt) is not None
    ]
    # Check for multiple options
    if len(modes) > 1:
        # Show which options were given
        optmodes = [f"--{o}" for o in modes]
        raise CapeValueError(f"Got: {' '.join(optmodes)}; can only use one")
    # Set mode
    mode = "raw" if (len(modes) == 0) else modes[0]
    # Run the case
    v = runner.collect_cutplanes(
        nsurf=nsurf,
        nbatch=nbatch,
        clean=clean,
        mode=mode,
        nmax=nmax,
        nproc=nproc)
    # Return code
    return IERR_OK, v


@CfdxCollectCutPlaneArgs.rst
def cape_triangulate_cutplane(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read instance
    runner, kw = read_runner(**kw)
    # Process args
    nproc = kw.get("nproc")
    nsurf = kw.get("nsurf")
    clean = kw.get("clean")
    # Run the case
    v = runner.triangulate_cutplane(nsurf, clean=clean, nproc=nproc)
    # Return code
    return IERR_OK, v


@CfdxSearchLargeArgs.rst
def cape_search_large(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Run the case
    v = manage.search_repo_large(**kw)
    # Return code
    return IERR_OK, v


@CfdxGetConfigArgs.rst
def cape_set_config(*a, **kw) -> Tuple[int, list]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Get value
    capeconfig.set_cape_opt(a[0], a[1], blend=kw.get("blend", False))
    # Return code
    return IERR_OK, a[1]


@CfdxSkeletonArgs.rst
def cape_skeleton(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxSkeletonArgs, *a, **kw)
    # Run the command
    v = cntl.SkeletonCases(**kw)
    # Return code
    return IERR_OK, v


@CfdxStartArgs.rst
def cape_start(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxStartArgs, *a, **kw)
    # Run the command
    v = cntl.SubmitJobs(nproc=1, **kw)
    # Return code
    return IERR_OK, v


@CfdxUnarchiveArgs.rst
def cape_unarchive(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxUnarchiveArgs, *a, **kw)
    # Run the command
    v = cntl.UnarchiveCases(**kw)
    # Return code
    return IERR_OK, v


@CfdxUnmarkArgs.rst
def cape_unmark(*a, **kw) -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Read *cntl*
    cntl, kw = read_cntl(CfdxUnmarkArgs, *a, **kw)
    # Run the command
    v = cntl.UnmarkCase(**kw)
    # Return code
    return IERR_OK, v


@CfdxUIArgs.rst
def cape_ui() -> Tuple[int, Any]:
    r"""Run ``%(title)s`` command

    %(description)s

    :Call:
        >>> ierr, v = %(name)s(*a, **kw)
    :Inputs:
        %(options)s
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *v*: **any**
            Output from API function
    """
    # Import user interface
    from .. import ui
    # Run code
    return ui.main(CfdxFrontDesk)


def read_cntl(cls: ArgReader, *a, **kw):
    r"""Read a CAPE run matrix control instance of appropriate class

    :Call:
        >>> cntl, parsed_kw = read_cntl(fname=None, solver=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of JSON file (or use most recent)
        *solver*: {``None``} | :class:`str`
            Solver module (or determine based on *fname*)
    :Outputs:
        *cntl*: :class:`cape.cfdx.cntl.Cntl`
            CAPE run matrix control instance (solver-specific)
    :Versions:
        * 2024-12-19 ``@ddalle``: v1.0
        * 2025-01-24 ``@ddalle``: v2.0; use module name instead of cls

    """
    # Get options
    fname = kw.pop("f", None)
    solver = kw.pop("solver", None)
    # Get module name if necessary
    if fname is None:
        # Find valid JSON files
        json_files = manage.find_json_solver()
        # Check for match
        if len(json_files) == 0:
            raise CapeValueError("Found no CAPE JSON files in repo")
        # Filter by solver if needed
        if solver is None:
            # No filter
            solver, fname = json_files[0]
        else:
            # Loop through matches
            for sj, fj in json_files:
                # Check solver
                if sj == solver:
                    # Found match
                    fname = fj
                    break
            else:
                raise CapeValueError("Found no CAPE JSON files in repo")
        # Report which file we're using!
        print(f"Using {solver} JSON file: {fname}")
    elif solver is None:
        # Determine solver
        solver = manage.identify_solver(fname)
    # Name of module
    modname = f"cape.{solver}.cntl"
    # Import it
    cntlmod = importlib.import_module(modname)
    # Instantiate
    cntl = cntlmod.Cntl(fname)
    # Instantiate args
    parser = cls(*a, **kw)
    parser.prog = parser._name.replace("cfdx-", f"{solver} ")
    # Record the JSON file if not given
    if "f" not in kw:
        parser.param_sequence.append(("f", fname))
    # Record it
    cntl.log_parser(parser)
    # Preprocess
    cntl.preprocess_kwargs(kw)
    # Output
    return cntl, kw


def read_runner(**kw) -> tuple:
    r"""Read a CAPE case runner instance to interact with CFD in `$PWD`

    :Call:
        >>> runner, kw = read_runner(**kw)
    :Inputs:
        *solver*: {``None``} | :class:`str`
            Solver module (e.g. ``"pyfun"``), or auto-determine
    :Outputs:
        *runner*: :class:`cape.cfdx.casecntl.CaseRunner`
            Case runner instance to control case in current folder
        *kw*: :class:`dict`
            Preprocessed keyword arguments
    :Versions:
        * 2024-12-30 ``@ddalle``: v1.0
        * 2025-01-24 ``@ddalle``: v2.0; use module name instead of cls
    """
    # Get options
    solver = kw.pop("solver", None)
    # Determine solver if necessary
    if solver is None:
        # Identify
        solver = manage.identify_case_solver()
        # Report usage
        print(f"Using solver module '{solver}'")
    # Name of module
    modname = f"cape.{solver}.casecntl"
    # Import it
    cntlmod = importlib.import_module(modname)
    # Instantiate
    runner = cntlmod.CaseRunner()
    # Output
    return runner, kw


# Name -> Function
CMD_DICT = {
    "1to2": cape_1to2,
    "apply": cape_apply,
    "approve": cape_approve,
    "archive": cape_archive,
    "batch": cape_batch,
    "collect-cutplane": cape_collect_cutplane,
    "collect-surf": cape_collect_surfdata,
    "check": cape_c,
    "check-db": cape_check_db,
    "check-fm": cape_check_fm,
    "check-ll": cape_check_ll,
    "check-triqfm": cape_check_triqfm,
    "clean": cape_clean,
    "defail": cape_defail,
    "dezombie": cape_dezombie,
    "edit-json": cape_edit,
    "exec": cape_exec,
    "extend": cape_extend,
    "extract": cape_extract_dex,
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
    "find-cases": cape_find,
    "find-json": cape_find_json,
    "find-large": cape_find_large,
    "get-config": cape_get_config,
    "open-pdf": cape_open_pdf,
    "post-file": cape_post_file,
    "qdel": cape_qdel,
    "receive-file": cape_receive_file,
    "report": cape_report,
    "rm": cape_rm,
    "run": cape_run,
    "search-large": cape_search_large,
    "set-config": cape_set_config,
    "skeleton": cape_skeleton,
    "start": cape_start,
    "triangulate-cutplane": cape_triangulate_cutplane,
    "ui": cape_ui,
    "unarchive": cape_unarchive,
    "unmark": cape_unmark,
}
# Invert *CMD_DICT*, Function Name -> Command Name
CMD_FUNCS = {v.__name__: k for k, v in CMD_DICT.items()}


# Template for each solver
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
    subparser.casecntl_mod = parser_cls._casecntl_mod
    # Parse args
    a, kw = subparser.get_a_kw()
    # Set default "solver"
    modname = parser_cls._cntl_mod
    if modname.startswith("cape.py"):
        kw.setdefault("solver", modname.split('.')[1])
    # Get function
    func = CMD_DICT.get(cmdname)
    # Call the function
    if func:
        # Use a try/except to catch user-input errors
        try:
            IERR, _ = func(*a, **kw)
            return IERR
        except (CapeError, ArgReadError) as e:
            # Print the error type
            sys.stderr.write(f"{e.__class__.__name__[4:]}:\n")
            # Now the error message
            for a in e.args:
                sys.stderr.write(f"    {a}\n")
            # End message and exit
            sys.stderr.flush()
            return IERR_RUNTIME
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            return IERR_INTERRUPT
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
    return main_template(CfdxFrontDesk, argv)


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
