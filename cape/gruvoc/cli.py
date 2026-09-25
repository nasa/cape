r"""
:mod:`cape.gruvoc.cli`: Command-line interface to ``gruvoc`` (executable)
==========================================================================

This module provides the :func:`main` function that is used by the
executable called ``gruvoc``.

"""

# Standard library modules
import os
import sys
from typing import Any, Optional, Tuple

# CAPE modules
from .errors import GruvocError, GruvocValueError
from .umesh import Umesh
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
class GruvocArgReader(ArgReader):
    # No attributes
    __slots__ = ()

    # Common options
    _optlist = (
        "help",
    )

    # No common aliases; note historical ``-h`` -> ``human`` is only an
    # alias for the *print* and *report* subcommands
    _optmap = {}

    # Option types
    _opttypes = {
        "add-cp": bool,
        "add-mach": bool,
        "flow": str,
        "h": bool,
        "help": bool,
        "human": bool,
        "i": str,
        "mapbc": str,
        "novol": bool,
        "nrows": int,
        "o": str,
        "smallvol": float,
        "tavg": str,
        "v": bool,
        "verbose": bool,
    }

    # Allowed types prior to conversion
    _rawopttypes = {
        "nrows": (int, str),
        "smallvol": (float, str),
    }

    # Conversion functions
    _optconverters = {
        "nrows": int,
        "smallvol": float,
    }

    # List of options that cannot take a "value"
    _optlist_noval = (
        "add-cp",
        "add-mach",
        "h",
        "help",
        "human",
        "novol",
        "v",
        "verbose",
    )

    # List of options that usually take a file name value
    _optlist_file = (
        "flow",
        "i",
        "mapbc",
        "mapbcfile",
        "o",
        "tavg",
    )

    # Description of each option
    _help_opt = {
        "add-cp": "Add pressure coefficient to surface state",
        "add-mach": "Add Mach number to surface state",
        "flow": "Name of FUN3D ``.flow`` file from which to read state",
        "help": "Print this help message and exit",
        "human": "Print human-readable mesh summary",
        "i": "Name of input mesh file",
        "mapbc": "Use ``.mapbc`` file to name surface components",
        "novol": "Delete volume cells before writing output mesh",
        "nrows": "Number of small-volume cells to include in report",
        "o": "Name of output mesh file",
        "smallvol": "Volume cutoff below which to report individual cells",
        "tavg": "Name of FUN3D ``.tavg`` file from which to read state",
        "verbose": "Write more verbose status messages",
    }

    # Name for value of select options in option descriptions
    _help_optarg = {
        "flow": "FLOWFILE",
        "i": "IFILE",
        "mapbc": "MAPBCFILE",
        "nrows": "N",
        "o": "OFILE",
        "smallvol": "VOLUME",
        "tavg": "TAVGFILE",
    }


# Settings for convert
class GruvocConvertArgs(GruvocArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "gruvoc convert"

    # Description
    _help_title = "Convert mesh from one format to another"

    # Additional options
    _optlist = (
        "add-cp",
        "add-mach",
        "flow",
        "i",
        "mapbc",
        "novol",
        "o",
        "tavg",
        "verbose",
    )

    # Command-specific aliases
    _optmap = {
        "mapbcfile": "mapbc",
        "v": "verbose",
    }

    # Alternate descriptions
    _help_opt = {
        "i": "Name of input file",
        "o": "Name of output file",
    }

    # Positional parameters
    _arglist = (
        "i",
        "o",
    )

    # Maximum number of positional arguments
    _nargmax = 2


# Settings for print and report
class GruvocPrintArgs(GruvocArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "gruvoc print"

    # Description
    _help_title = "Print summary of a surface or volume mesh"

    # Additional options
    _optlist = (
        "human",
        "i",
    )

    # Command-specific aliases; here historical ``-h`` means "human"
    _optmap = {
        "h": "human",
    }

    # Positional parameters
    _arglist = (
        "i",
    )

    # Maximum number of positional arguments
    _nargmax = 1


# Settings for small-vols and report-small-vols
class GruvocSmallVolsArgs(GruvocArgReader):
    # No attributes
    __slots__ = ()

    # Name of function
    _name = "gruvoc small-vols"

    # Description
    _help_title = "Report cells with smallest volumes"

    # Additional options
    _optlist = (
        "i",
        "mapbc",
        "nrows",
        "smallvol",
    )

    # Command-specific aliases
    _optmap = {
        "mapbcfile": "mapbc",
        "n": "nrows",
    }

    # Alternate descriptions
    _help_opt = {
        "i": "Name of input volume mesh file",
    }

    # Positional parameters
    _arglist = (
        "i",
    )

    # Maximum number of positional arguments
    _nargmax = 1

    # Defaults
    _rc = {
        "nrows": 25,
    }


# Argument settings for main gruvoc interface
class GruvocFrontDesk(GruvocArgReader):
    # No attributes
    __slots__ = ()

    # Name of executable
    _name = "gruvoc"

    # Description of executable
    _help_title = "Unstructured volume mesh interface from CAPE team"

    # Longer description
    _help_description = (
        "Convert, move, or analyze unstructured meshes with tets,\n"
        "pyramids, prisms, and/or hexs.")

    # List of available options (in any subcommand, including any
    # spellings of command-specific aliases)
    _optlist = (
        "add-cp",
        "add-mach",
        "flow",
        "h",
        "human",
        "i",
        "mapbc",
        "mapbcfile",
        "n",
        "novol",
        "nrows",
        "o",
        "smallvol",
        "tavg",
        "v",
        "verbose",
    )

    # No aliases; parse each spelling as written
    _optmap = {}

    # List of sub-commands
    _cmdlist = (
        "help",
        "convert",
        "print",
        "report",
        "report-small-vols",
        "small-vols",
    )

    # Alternate command names
    _cmdmap = {
        "small_vols": "small-vols",
        "report_small_vols": "report-small-vols",
    }

    # Subparsers
    _cmdparsers = {
        "convert": GruvocConvertArgs,
        "print": GruvocPrintArgs,
        "report": GruvocPrintArgs,
        "report-small-vols": GruvocSmallVolsArgs,
        "small-vols": GruvocSmallVolsArgs,
    }

    # Description of sub-commands
    _help_cmd = {
        "help": "Display help message and exit",
        "report": "Print summary of a surface or volume mesh",
        "report-small-vols": "Report cells with smallest volumes",
    }

    # List of options for --help
    _help_optlist = (
        "help",
    )


@GruvocConvertArgs.rst
def gruvoc_convert(*a, **kw) -> Tuple[int, Any]:
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
    # Get file names, either from args or options
    ifile = a[0] if len(a) > 0 else kw.pop("i", None)
    ofile = a[1] if len(a) > 1 else kw.pop("o", None)
    # Check for both files
    if ifile is None:
        raise GruvocValueError("No input mesh file provided to convert")
    if ofile is None:
        raise GruvocValueError("No output mesh file provided to convert")
    # Read mesh
    mesh = Umesh(ifile, mapbc=kw.get("mapbc"))
    # Read FUN3D .flow file if appropriate
    flowfile = kw.get("flow")
    if flowfile:
        mesh.read_fun3d_flow(flowfile)
    tavgfile = kw.get("tavg")
    if tavgfile:
        mesh.read_fun3d_tavg(tavgfile)
    # Delete volume
    if kw.get("novol"):
        mesh.remove_volume()
    # Post-read options
    if kw.get("add-mach"):
        mesh.add_mach()
    if kw.get("add-cp"):
        mesh.add_cp()
    # Write output
    mesh.write(ofile, v=kw.get("verbose", False))
    # Return code
    return IERR_OK, mesh


@GruvocPrintArgs.rst
def gruvoc_print_summary(*a, **kw) -> Tuple[int, Any]:
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
    # Get input file name, either from arg or option
    ifile = a[0] if len(a) else kw.pop("i", None)
    # Check for a file
    if ifile is None:
        raise GruvocValueError("No input mesh file provided to print")
    # Get summar format option
    human = kw.get("human", False)
    # Read mesh (meta-mode)
    mesh = Umesh(ifile, meta=True)
    # Write summary
    mesh.print_summary(h=human)
    # Return code
    return IERR_OK, mesh


@GruvocSmallVolsArgs.rst
def gruvoc_small_vols(*a, **kw) -> Tuple[int, Any]:
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
    # Get input file name, either from arg or option
    ifile = a[0] if len(a) else kw.pop("i", None)
    # Check for a file
    if ifile is None:
        raise GruvocValueError("No input mesh file provided to small-vols")
    # Get other options
    nrows = kw.get("nrows", 25)
    mapbcfile = kw.get("mapbc")
    smallvol = kw.get("smallvol")
    # Read mesh
    mesh = Umesh(ifile, mapbc=mapbcfile)
    # Generate report
    v = mesh.report_small_cells(smallvol, nrows=nrows)
    # Return code
    return IERR_OK, v


# Name -> Function
CMD_DICT = {
    "convert": gruvoc_convert,
    "print": gruvoc_print_summary,
    "report": gruvoc_print_summary,
    "report-small-vols": gruvoc_small_vols,
    "small-vols": gruvoc_small_vols,
}
# Invert *CMD_DICT*, Function Name -> Command Name
CMD_FUNCS = {v.__name__: k for k, v in CMD_DICT.items()}


# Template for each front desk
def main_template(
        parser_cls: GruvocFrontDesk,
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
    # Check for ``--help``
    if subparser.show_help("help"):
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
        except (GruvocError, CapeError, ArgReadError, OptdictError) as e:
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
    r"""Main interface to ``gruvoc``

    :Call:
        >>> main()
    :Versions:
        * 2025-04-04 ``@ddalle``: v1.0
        * 2026-09-22 ``@ddalle``: v2.0; use ``argread``
    """
    return main_template(GruvocFrontDesk, argv)


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
