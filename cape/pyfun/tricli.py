r"""
This module includes functions to interface with triangulated surface
files and provides interfaces accessible from the command line.

Many of these functions perform conversions, for example
:func:`uh3d2tri` reads a UH3D file and converts it to Cart3D ``.tri``
format.
"""

# Standard library modules
import os
import sys

# Third-party modules
import numpy as np

# Local imprts
from .. import argread
from .. import text as textutils
from .plt import Plt2Triq


# Template help messages
_help_help = r"""-h, --help
        Display this help message and exit"""
_help_config = r"""-c CONFIGFILE
        Use file *CONFIGFILE* to map component ID numbers; guess type
        based on file name

    --xml XML
        Use file *XML* as config file with XML format

    --json JSON
        Use file *JSON* as JSON-style surface config file

    --mixsur MIXSUR
        Use file *MIXSUR* to label surfaces assuming ``mixsur`` or
        ``usurp`` input file format"""

_help = {
    "config": _help_config,
    "help": _help_help,
}

# Help message for "uh3d2tri"
HELP_PLT2TRIQ = r"""
``pyfun-plt2triq``: Convert FUN3D Tecplot file to Cart3D tri file
======================================================================

Convert a Tecplot ``.plt`` file from FUN3D 

:Call:

    .. code-block:: console
    
        $ pyfun-plt2triq PLT [TRIQ] [OPTIONS]
        $ pyfun-plt2triq -i PLT [-o TRIQ] [OPITONS] 

:Inputs:
    * *PLT*: Name of FUN3D Tecplot ``.plt`` file
    * *TRIQ*: Name of output ``.triq`` file
    
:Options:
    %(help)s
        
    -i PLT
        Use *PLT* as input file
        
    -o TRIQ
        Use *TRIQ* as name of created output files
        
    --mach, -m MINF
        Use *MINF* to scale skin friction coefficients
        
    --no-triload
        Use all state variables in order instead of extracting preferred
        state variables intended for ``triloadCmd``
        
    --no-avg
        Do not use time average, even if available
        
    --rms
        Write RMS of each variable instead of nominal/average value
    
If the name of the output file is not specified, it will just add ``triq`` as
the extension to the input (replacing ``.plt`` if possible).

:Versions:
    * 2016-12-19 ``@ddalle``: Version 1.0
    * 2021-10-14 ``@ddalle``: Version 1.1; moved into :mod:`cape.pyfun`
""" % _help


def plt2triq(*a, **kw):
    r"""Convert FUN3D surface ``.plt`` to Cart3D ``.triq`` format
    
    :Call:
        >>> plt2triq(fplt, **kw)
        >>> plt2triq(fplt, ftriq, **kw)
        >>> plt2triq(i=fplt, o=ftriq, **kw)
    :Inputs:
        *fplt*: :class:`str`
            Name of input file Tecplot file (can be ASCII)
        *ftriq*: {``None``} | :class:`str`
            Name of annotated triangulation ``.triq`` file to create;
            defaults to *fplt* with the ``.plt`` replaced by ``.triq``
        *mach*: {``1.0``} | positive :class:`float`
            Freestream Mach number for skin friction coeff conversion
        *triload*: {``True``} | ``False``
            Whether or not to write a triq tailored for ``triloadCmd``
        *avg*: {``True``} | ``False``
            Use time-averaged states if available
        *rms*: ``True`` | {``False``}
            Use root-mean-square variation instead of nominal value
    :See also:
        * :func:`cape.pyfun.plt.Plt2Triq`
    :Versions:
        * 2016-12-19 ``@ddalle``: Version 1.0; :func:`Plt2Triq`
        * 2021-10-01 ``@ddalle``: Version 2.0
    """
    # Get input file name
    fplt = _get_i(*a, **kw)
    # Get output file name
    ftriq = _get_o(fplt, "plt", "triq", *a, **kw)
    # Convert file
    Plt2Triq(fplt, ftriq=ftriq, **kw)


# CLI functions
def main_plt2triq():
    r"""CLI for :func:`plt2triq`

    :Call:
        >>> main_plt2triq()
    :Versions:
        * 2021-10-14 ``@ddalle``: Version 1.0
    """
    _main(plt2triq, HELP_PLT2TRIQ)


def _main(func, doc):
    r"""Command-line interface template

    :Call:
        >>> _main(func, doc)
    :Inputs:
        *func*: **callable**
            API function to call after processing args
        *doc*: :class:`str`
            Docstring to print with ``-h``
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Process the command-line interface inputs.
    a, kw = argread.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h', False) or kw.get("help", False):
        print(textutils.markdown(doc))
        return
    # Run the main function.
    func(*a, **kw)
    

# Process sys.argv
def _get_i(*a, **kw):
    r"""Process input file name

    :Call:
        >>> fname_in = _get_i(*a, **kw)
        >>> fname_in = _get_i(fname)
        >>> fname_in = _get_i(i=None)
    :Inputs:
        *a[0]*: :class:`str`
            Input file name specified as first positional arg
        *i*: :class:`str`
            Input file name as kwarg; supersedes *a*
    :Outputs:
        *fname_in*: :class:`str`
            Input file name
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Get the input file name
    if len(a) == 0:
        # Defaults
        fname_in = None
    else:
        # Use the first general input
        fname_in = a[0]
    # Prioritize a "-i" input
    fname_in = kw.get('i', fname_in)
    # Must have a file name.
    if fname_in is None:
        # Required input.
        raise ValueError("Required at least 1 arg or 'i' kwarg")
    # Output
    return fname_in


def _get_o(fname_in, ext1, ext2, *a, **kw):
    r"""Process output file name

    :Call:
        >>> fname_out = _get_o(fname_in, ext1, ext2, *a, **kw)
    :Inputs:
        *fname_in*: :class:`str`
            Input file name
        *ext1*: :class:`str`
            Expected file extension for *fname_in*
        *ext2*: :class:`str`
            Default file extension for *fname_out*
        *a[1]*: :class:`str`
            Output file name specified as first positional arg
        *o*: :class:`str`
            Output file name as kwarg; supersedes *a*
    :Outputs:
        *fname_out*: :class:`str`
            Output file name
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Strip *ext1* as starter for default
    if fname_in.endswith(ext1):
        # Strip expected file extension
        fname_out = fname_in[:-len(ext1)]
    else:
        # Use current file name since extension not found
        fname_out = fname_in + "."
    # Add supplied *ext2* extension for output
    fname_out = fname_out + ext2
    # Get the output file name
    if len(a) >= 2:
        fname_out = a[1]
    # Prioritize a "-o" input.
    fname_out = kw.get('o', fname_out)
    # Output
    return fname_out

