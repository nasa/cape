r"""
:mod:`cape.dkit.cli`: Command-line interface to ``dkit`` (executable)
=======================================================================

This module provides the :func:`main` function that is used by the
executable called ``dkit``.

"""

# Standard library modules
import os
import sys
from typing import Any, Optional, Tuple

# CAPE modules
from . import quickstart
from . import vendorutils
from . import writedb
from ..argread import ArgReader, ArgReadError
from ..errors import CapeError
from ..optdict.opterror import OptdictError


# Constants
IERR_OK = 0
IERR_INTERRUPT = 2
IERR_CMD = 16
IERR_OPT = 32
IERR_RUNTIME = 128


# Common argument settings
class DkitArgReader(ArgReader):
    # No attributes
    __slots__ = ()

    # Common options
    _optlist = (
        "h",
    )

    # Common aliases
    _optmap = {
        "F": "force-all",
        "dependencies": "reqs",
        "force_all": "force-all",
        "func": "write_func",
        "help": "h",
        "requirements": "reqs",
        "t": "target",
        "write-func": "write_func",
    }

    # Option types
    _opttypes = {
        "check": bool,
        "cwd": str,
        "f": (bool, str),
        "force": bool,
        "force-all": bool,
        "h": bool,
        "install": bool,
        "json": str,
        "prefix": str,
        "reqs": bool,
        "target": str,
        "title": str,
        "toml": str,
        "where": str,
        "write": bool,
        "write_func": str,
    }

    # List of options that cannot take a "value"
    _optlist_noval = (
        "check",
        "force",
        "force-all",
        "h",
        "install",
        "reqs",
        "write",
    )

    # List of options that usually take a file name value
    _optlist_file = (
        "json",
        "toml",
    )

    # Description of each option
    _help_opt = {
        "check": "List packages to vendorize but don't install them",
        "cwd": "Location from which to search for packages",
        "force": (
            "Overwrite any existing database files (only for *MODNAMES*)"),
        "force-all": (
            "Overwrite all database files including added dependencies"),
        "h": "Print this help message and exit",
        "install": "List packages to vendorize but don't install them",
        "json": "Search for vendorize files called *FJSON*",
        "prefix": "Specify prefix which may be left off of *MODNAMES*",
        "reqs": "Don't read requirements; just process *MODNAMES*",
        "target": "Only vendorize in parent packages matching *REGEX*",
        "title": "Use *TITLE* as one-line description for the package",
        "toml": "Search for vendorize files called *FTOML*",
        "where": "Create package in folder *WHERE*",
        "write": "Don't actually write databases (just print dependencies)",
        "write_func": "Function name in modules to process datakit files",
    }

    # Name for value of select options in option descriptions
    _help_optarg = {
        "cwd": "WHERE",
        "json": "FJSON",
        "prefix": "PREFIX",
        "target": "TARGET",
        "title": "TITLE",
        "toml": "FTOML",
        "where": "WHERE",
        "write_func": "FUNC",
    }

    # List of options that should be shown as negative in help
    _help_opt_negative = (
        "install",
        "reqs",
        "write",
    )


# Settings for quickstart
class DkitQuickstartArgs(DkitArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "dkit quickstart"

    # Description
    _help_title = "Create a template DataKit package"

    # Additional options
    _optlist = (
        "target",
        "title",
        "where",
    )

    # Alternate descriptions
    _help_opt = {
        "pkg": "Name of Python package relative to *WHERE*",
        "target": "Use *TARGET* as a prefix to the package name *PKG*",
    }

    # Positional parameters
    _arglist = (
        "pkg",
        "where",
    )


# Settings for vendorize
class DkitVendorizeArgs(DkitArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "dkit vendorize"

    # Description
    _help_title = "Install local copies of packages"

    # Additional options
    _optlist = (
        "check",
        "cwd",
        "install",
        "json",
        "target",
        "toml",
    )

    # Command-specific aliases
    _optmap = {
        "f": "json",
        "where": "cwd",
    }

    # Alternate descriptions
    _help_opt = {
        "pkg": "Vendorize each package matching this regular expression",
    }

    # Name for value of select options in option descriptions
    _help_optarg = {
        "target": "REGEX",
    }

    # Positional parameters
    _arglist = (
        "pkg",
    )


# Settings for writedb
class DkitWriteDBArgs(DkitArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "dkit writedb"

    # Description
    _help_title = "Process raw data into datakit files"

    # Additional options
    _optlist = (
        "force",
        "force-all",
        "prefix",
        "reqs",
        "write",
        "write_func",
    )

    # Command-specific aliases
    _optmap = {
        "f": "force",
    }

    # Alternate descriptions
    _help_opt = {
        "modnames": "Name(s) of modules to process, e.g. ``db0001``",
    }

    # Positional parameters
    _arglist = (
        "modnames",
    )

    # Defaults
    _rc = {
        "reqs": True,
        "write": True,
    }


# Argument settings for main dkit interface
class DkitFrontDesk(DkitArgReader):
    # No attributes
    __slots__ = ()

    # Name of executable
    _name = "dkit"

    # Description of executable
    _help_title = "Command-line interface to datakit tools"

    # Longer description
    _help_description = (
        "Perform actions on a DataKit package or package collection\n"
        "from the command-line interface.")

    # List of available options (in any subcommand)
    _optlist = (
        "check",
        "cwd",
        "f",
        "force",
        "force-all",
        "install",
        "json",
        "prefix",
        "reqs",
        "target",
        "title",
        "toml",
        "where",
        "write",
        "write_func",
    )

    # List of sub-commands
    _cmdlist = (
        "help",
        "quickstart",
        "vendorize",
        "writedb",
    )

    # Alternate command names
    _cmdmap = {
        "quick_start": "quickstart",
        "write-db": "writedb",
        "write_db": "writedb",
    }

    # Subparsers
    _cmdparsers = {
        "quickstart": DkitQuickstartArgs,
        "vendorize": DkitVendorizeArgs,
        "writedb": DkitWriteDBArgs,
    }

    # Description of sub-commands
    _help_cmd = {
        "help": "Display help message and exit",
    }

    # List of options for --help
    _help_optlist = (
        "h",
    )


@DkitQuickstartArgs.rst
def dkit_quickstart(*a, **kw) -> Tuple[int, Any]:
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
    # Run quickstart
    v = quickstart.quickstart(*a, **kw)
    # Return code
    return IERR_OK, v


@DkitVendorizeArgs.rst
def dkit_vendorize(*a, **kw) -> Tuple[int, Any]:
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
    # Run vendorize
    v = vendorutils.vendorize_repo(*a, **kw)
    # Return code
    return IERR_OK, v


@DkitWriteDBArgs.rst
def dkit_writedb(*a, **kw) -> Tuple[int, Any]:
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
    # Run writedb
    v = writedb.write_dbs(*a, **kw)
    # Return code
    return IERR_OK, v


# Name -> Function
CMD_DICT = {
    "quickstart": dkit_quickstart,
    "vendorize": dkit_vendorize,
    "writedb": dkit_writedb,
}
# Invert *CMD_DICT*, Function Name -> Command Name
CMD_FUNCS = {v.__name__: k for k, v in CMD_DICT.items()}


# Template for each front desk
def main_template(
        parser_cls: DkitFrontDesk,
        argv: Optional[list] = None) -> int:
    # Create parser
    parser = parser_cls()
    # Use sys.argv if necessary
    argv = _get_argv(argv)
    # Identify subcommand
    try:
        cmdname, subparser, ierr = parser.fullparse_check(argv)
    except ArgReadError as e:
        # Print the error type
        sys.stderr.write(f"{e.__class__.__name__}:\n")
        # Now the error message
        for a in e.args:
            sys.stderr.write(f"    {a}\n")
        # End message and exit
        sys.stderr.flush()
        return IERR_RUNTIME
    # Check for errors
    if ierr:
        return IERR_OPT
    # Check for valid command name or other front-desk help triggers
    if parser.help_frontdesk(cmdname):
        return IERR_OK
    # Check for ``-h``
    if subparser.show_help("h"):
        return IERR_OK
    # Parse args
    a, kw = subparser.get_a_kw()
    # Get function
    func = CMD_DICT.get(cmdname)
    # Call the function
    if func:
        # Use a try/except to catch user-input errors
        try:
            IERR, _ = func(*a, **kw)
            return IERR
        except (CapeError, ArgReadError, OptdictError) as e:
            # Print the error type
            sys.stderr.write(f"{e.__class__.__name__}:\n")
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


# Primary interface
def main(argv: Optional[list] = None) -> int:
    r"""Main interface to ``dkit``

    :Call:
        >>> main()
    :Versions:
        * 2021-08-24 ``@ddalle``: v1.0
        * 2026-09-22 ``@ddalle``: v2.0; use ``argread``
    """
    return main_template(DkitFrontDesk, argv)


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
