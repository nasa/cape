r"""
:mod:`cape.teccli`: Command-line interfaces to Tecplot(R)
==============================================================

This module provides a function to easily export an image using a given
Tecplot(R) layou (``.lay``) file.
"""

# Standard library modules
from typing import Optional

# Local imprts
from .argread import ArgReader
from .argread.clitext import compile_rst
from .filecntl.tecfile import ExportLayout


# Arguments
class CapeTecArgParser(ArgReader):
    # No attributes
    __slots__ = ()

    # Allowed options
    _optlist = (
        "clean",
        "ext",
        "help",
        "layout",
        "o",
        "verbose",
        "width",
    )

    # Aliases
    _optmap = {
        "fmt": "ext",
        "format": "ext",
        "h": "help",
        "lay": "layout",
        "output": "o",
        "v": "verbose",
        "w": "width",
    }

    # Positional parameters
    _arglist = (
        "layout",
        "o",
    )

    # Required options/args
    _nargmin = 1

    # Converters
    _optconverters = {
        "w": int,
    }

    # No-value options
    _optlist_noval = (
        "clean",
        "help",
        "verbose",
    )

    # Default
    _rc = {
        "ext": "PNG",
    }

    # Primary aspects of function
    _name = "cape-tec"
    _help_title = "Export image from Tecplot(R) layout file"

    # Descriptions
    _help_opt = {
        "clean": "Delete macro after export (use ``--no-clean`` to suppress)",
        "ext": "Export format",
        "help": "Display this help message and exit",
        "layout": "Name of Tecplot(R) layout file to use",
        "o": "Output file (default based on *layout* with changed extension)",
        "verbose": "Increase verbosity during process",
        "width": "Image width, in pixels",
    }

    # Arguments for options in help message
    _help_optarg = {
        "ext": "EXT",
        "layout": "LAY",
        "o": "FNAME",
        "width": "WIDTH",
    }


# CLI functions
def export_layout(argv: Optional[list] = None) -> int:
    r"""CLI for Tecplot(R) layout -> image export

    :Call:
        >>> ierr = export_layout(argv=None)
    :Inputs:
        *argv*: {``None``} | :class:`list`\ [:class:`str`]
            List of CLI args (else use ``sys.argv``)
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-11-15 ``@ddalle``: v1.0
    """
    # Create parser
    parser = CapeTecArgParser()
    # Parse CLI text
    parser.parse(argv)
    # Check for help message
    if parser.get("help", False):
        print(compile_rst(parser.genr8_help()))
        return 0
    # Get all named options
    kw = parser.get_kwargs()
    # Get main options
    lay = kw.pop("layout", "layout.lay")
    fname = kw.pop("o", None)
    ext = kw.pop("ext", "PNG")
    # Call main function
    ExportLayout(lay, fname, ext, **kw)
    # Return
    return 0

