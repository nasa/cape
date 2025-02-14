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
from .filecntl import tecfile


# Arguments
class _TecArgParser(ArgReader):
    # No attributes
    __slots__ = ()

    # Aliases
    _optmap = {
        "fmt": "ext",
        "format": "ext",
        "h": "help",
        "lay": "layout",
        "output": "o",
        "sample": "supersample",
        "super": "supersample",
        "v": "verbose",
        "w": "width",
    }

    # Converters
    _optconverters = {
        "supersample": int,
        "width": int,
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

    # Descriptions
    _help_opt = {
        "clean": "Delete macro after export (use ``--no-clean`` to suppress)",
        "ext": "Export format",
        "help": "Display this help message and exit",
        "layout": "Name of Tecplot(R) layout file to use",
        "o": "Output file (default based on *layout* with changed extension)",
        "plt": "Name of Tecplot(R) PLT file to write",
        "supersample": "Number of supersamples to make while anti-aliasing",
        "szplt": "Name of Tecplot(R) SZPLT file to convert",
        "verbose": "Increase verbosity during process",
        "width": "Image width, in pixels",
    }

    # Arguments for options in help message
    _help_optarg = {
        "ext": "EXT",
        "layout": "LAY",
        "o": "FNAME",
        "plt": "PLTFILE",
        "supersample": "S",
        "szplt": "SZPLTFILE",
        "width": "WIDTH",
    }


# Arguments for ``cape-tec``
class CapeTecArgParser(_TecArgParser):
    # No attributes
    __slots__ = ()

    # Allowed options
    _optlist = (
        "clean",
        "ext",
        "help",
        "layout",
        "o",
        "supersample",
        "verbose",
        "width",
    )

    # Positional parameters
    _arglist = (
        "layout",
        "o",
    )

    # Defaults
    _rc = {
        "supersample": 3,
    }

    # Required options/args
    _nargmin = 1

    # Primary aspects of function
    _name = "cape-tec"
    _help_title = "Export image from Tecplot(R) layout file"


# Arguments for ``cape-szplt2plt``
class CapeSzplt2PltArgParser(_TecArgParser):
    # No attributes
    __slots__ = ()

    # Allowed options
    _optlist = (
        "antialias",
        "clean",
        "help",
        "o",
        "plt",
        "szplt",
        "verbose",
    )

    # Positional parameters
    _arglist = (
        "szplt",
        "o",
    )

    # Required options/args
    _nargmin = 1

    # Primary aspects
    _name = "cape-szplt2plt"
    _help_title = "Convert Tecplot(R) SZPLT -> PLT format"


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
    tecfile.ExportLayout(lay, fname, ext, **kw)
    # Return
    return 0


# CLI functions
def convert_szplt(argv: Optional[list] = None) -> int:
    r"""CLI for Tecplot(R) SZPLT -> PLT

    :Call:
        >>> ierr = convert_szplt(argv=None)
    :Inputs:
        *argv*: {``None``} | :class:`list`\ [:class:`str`]
            List of CLI args (else use ``sys.argv``)
    :Outputs:
        *ierr*: :class:`int`
            Return code
    :Versions:
        * 2024-12-03 ``@ddalle``: v1.0
    """
    # Create parser
    parser = CapeSzplt2PltArgParser()
    # Parse CLI text
    parser.parse(argv)
    # Check for help message
    if parser.get("help", False):
        print(compile_rst(parser.genr8_help()))
        return 0
    # Get all named options
    kw = parser.get_kwargs()
    # Get main options
    fszplt = kw.pop("szplt", "tecplot.szplt")
    fplt = kw.pop("o", None)
    # Call main function
    tecfile.convert_szplt(fszplt, fplt, **kw)
    # Return
    return 0

