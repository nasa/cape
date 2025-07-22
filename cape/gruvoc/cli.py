
# Local imports
from .errors import GruvocError
from .umesh import Umesh
from ..argread import ArgReader
from ..argread.clitext import compile_rst


# Return codes
IERR_OK = 0
IERR_CMD = 16
IERR_ARGS = 32
IERR_FILE_NOT_FOUND = 128


# Help message
HELP_GRUVOC = r"""
``gruvoc``: Unstructured volume mesh interface from CAPE team
===============================================================

Convert, move, or analyze unstructured meshes with tets, pyramids,
prisms, and/or hexs.

:Usage:
    .. code-block:: console

        $ gruvoc CMD [OPTIONS]

:Inputs:
    * *CMD*: name of command to run

    Available commands are:

    ==================  ===========================================
    Command             Description
    ==================  ===========================================
    ``convert``         Convert mesh from one format to another
    ==================  ===========================================
"""


HELP_CONVERT = r"""
``gruvoc-convert``: Convert surface or volume mesh format
===========================================================

This command reads a mesh from one file and writes it to another file,
probably using a different file format.

:Usage:
    .. code-block:: console

        $ gruvoc convert IFILE OFILE [OPTIONS]

:Inputs:
    * *IFILE*: Name of input file
    * *OFILE*: Name of output file

:Options:
    -h, --help
        Display this help message and exit

    --mapbc MAPBCFILE
        Use ``.mapbc`` file to name surface components

    -v, --verbose
        Write more verbose status messages

    -i IFILE
        Use *IFILE* as input file

    -o OFILE
        Use *OFILE* as output file
"""

HELP_DICT = {
    "convert": HELP_CONVERT,
}


# Customized CLI parser
class GruvocArgParser(ArgReader):
    # No attributes
    __slots__ = ()

    # Aliases
    _nargmin = 1


class GruvocConvertArgParser(ArgReader):
    __slots__ = ()

    _optlist = (
        "i",
        "o",
        "add-cp",
        "add-mach",
        "flow",
        "mapbc",
        "novol",
        "tavg",
        "verbose",
    )

    _optmap = {
        "v": "verbose",
        "mapbcfile": "mapbc",
    }

    _arglist = (
        "i",
        "o",
    )

    _opttypes = {
        "novol": bool,
        "verbose": bool,
    }

    _nargmax = 2


class GruvocPrintParser(ArgReader):
    __slots__ = ()

    _optlist = (
        "i",
        "h",
    )

    _optmap = {
        "human": "h",
    }

    _arglist = (
        "i",
    )


class GruvocSmallVolParser(ArgReader):
    __slots__ = ()

    _optlist = (
        "i",
        "mapbc",
        "nrows",
        "smallvol",
    )

    _optmap = {
        "mapbcfile": "mapbc",
        "n": "nrows",
    }

    _arglist = (
        "i",
    )

    _optconverters = {
        "smallvol": float,
    }

    _opttypes = {
        "nrows": int,
    }


# Convert a mesh
def gruvoc_convert(*a, **kw) -> int:
    # Parse args
    parser = GruvocConvertArgParser(*a, **kw)
    kw = parser.get_kwargs()
    # Get options
    ifile = kw.get("i")
    ofile = kw.get("o")
    mapbcfile = kw.get("mapbc")
    verbose = kw.get("verbose")
    novol = kw.get("novol")
    flowfile = kw.get("flow")
    tavgfile = kw.get("tavg")
    # Read mesh
    mesh = Umesh(ifile, mapbc=mapbcfile)
    # Read FUN3D .flow file if appropriate
    if flowfile:
        mesh.read_fun3d_flow(flowfile)
    if tavgfile:
        mesh.read_fun3d_tavg(tavgfile)
    # Delete volume
    if novol:
        mesh.remove_volume()
    # Post-read options
    if kw.get("add-mach"):
        mesh.add_mach()
    if kw.get("add-cp"):
        mesh.add_cp()
    # Write output
    mesh.write(ofile, v=verbose)
    # Output
    return 0


# Summary
def gruvoc_print_summary(*a, **kw) -> int:
    # Parse args
    parser = GruvocPrintParser(*a, **kw)
    kw = parser.get_kwargs()
    # Get options
    ifile = kw.get("i")
    human = kw.get("h")
    # REad mesh (meta-mode)
    mesh = Umesh(ifile, meta=True)
    # Write summary
    mesh.print_summary(h=human)
    # Output
    return 0


# Report small volumes
def gruvoc_small_vols(*a, **kw) -> int:
    # Parse args
    parser = GruvocSmallVolParser(*a, **kw)
    kw = parser.get_kwargs()
    # Get options
    ifile = kw.get("i")
    nrows = kw.get("nrows", 25)
    mapbcfile = kw.get("mapbc")
    smallvol = kw.get("smallvol")
    # Read mesh
    mesh = Umesh(ifile, mapbc=mapbcfile)
    # Generate report
    mesh.report_small_cells(smallvol, nrows=nrows)
    # Return code
    return 0


# Command dictionary
CMD_DICT = {
    "convert": gruvoc_convert,
    "print": gruvoc_print_summary,
    "report": gruvoc_print_summary,
    "small-vols": gruvoc_small_vols,
    "report-small-vols": gruvoc_small_vols,
}


def main() -> int:
    # Create parser
    parser = GruvocArgParser()
    # Parse args
    a, kw = parser.parse()
    kw.pop("__replaced__", None)
    # Check for no commands
    if len(a) == 0:
        print(compile_rst(HELP_GRUVOC))
        return 0
    # Get command name
    cmdname = a[0]
    # Get function
    func = CMD_DICT.get(cmdname)
    # Check it
    if func is None:
        # Unrecognized function
        print("Unexpected command '%s'" % cmdname)
        print("Options are: " + " | ".join(list(CMD_DICT.keys())))
        return IERR_CMD
    # Check for "help" option
    if kw.get("help", False):
        # Get help message for this command; default to main help
        msg = HELP_DICT.get(cmdname, HELP_GRUVOC)
        print(compile_rst(msg))
        return 0
    # Run function
    try:
        ierr = func(*a[1:], **kw)
    except GruvocError as err:
        print(f"{err.__class__.__name__}:")
        print(f"  {err}")
        return 1
    # Convert None -> 0
    ierr = IERR_OK if ierr is None else ierr
    # Normal exit
    return ierr
