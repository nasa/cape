r"""
:mod:`cape.tricli`: Interfaces to ``.tri`` and related files
==============================================================

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
from . import argread
from . import text as textutils
from .config import ConfigXML, ConfigMIXSUR, ConfigJSON
from .plt import Plt
from .tri import Tri


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
HELP_UH3D2TRI = r"""
``cape-uh3d2tri``: Convert UH3D triangulation to Cart3D format
===============================================================

Convert a ``.uh3d`` file to a Cart3D triangulation format.

:Usage:

    .. code-block:: console
    
        $ cape-uh3d2tri UH3D [OPTIONS]
        $ cape-uh3d2tri UH3D TRI [OPTIONS]
        $ cape-uh3d2tri [OPTIONS]

:Inputs:
    * *UH3D*: Name of input '.uh3d' file
    * *TRI*: Name of output '.tri' file
    
:Options:
    %(help)s
        
    -i UH3D
        Use *UH3D* as input file
        
    -o TRI
        Use *TRI* as name of created output file
       
    %(config)s

    --ascii
        Write *TRI* as an ASCII file (default)
        
    --binary, --bin
        Write *TRI* as an unformatted Fortran binary file
        
    --byteorder BO, --endian BO
        Override system byte order using either 'big' or 'little'
        
    --bytecount PREC
        Use a *PREC* of 4 for single- or 8 for double-precision
        
    --xtol XTOL
        Truncate nodal coordinates within *XTOL* of x=0 plane to zero
        
    --ytol YTOL
        Truncate nodal coordinates within *YTOL* of y=0 plane to zero
        
    --ztol ZTOL
        Truncate nodal coordinates within *ZTOL* of z=0 plane to zero

    --dx DX
        Translate all nodes by *DX* in *x* direction

    --dy DY
        Translate all nodes by *DY* in *y* direction

    --dz DZ
        Translate all nodes by *DZ* in *z* direction
    
If the name of the output file is not specified, it will just add '.tri'
as the extension to the input (deleting '.uh3d' if possible).

:Versions:
    * 2014-06-12 ``@ddalle``: Version 1.0
    * 2015-10-09 ``@ddalle``: Version 1.1
        - Add tolerances and ``Config.xml`` processing
        - Add *dx*, *dy*, *dz* translation options
""" % _help

HELP_TRI2UH3D = r"""
``cape-tri2uh3d``: Convert Cart3D Triangulation to UH3D Format
===================================================================

Convert a Cart3D triangulation ``.tri`` file to a UH3D file.  The most
common purpose for this task is to inspect triangulations with moving
bodies with alternative software such as ANSA.

:Usage:
    .. code-block:: console
    
        $ cape-tri2uh3d TRI [OPTIONS]
        $ cape-tri2uh3d TRI UH3D [OPTIONS]
        $ cape-tri2uh3d -i TRI [-o UH3D] [OPTIONS]

:Inputs:
    * *TRI*: Name of output ``.tri`` file
    * *UH3D*: Name of input ``.uh3d`` file

:Options:
    %(help)s

    -i TRI
        Use *TRI* as name of created output file

    -o UH3D
        Use *UH3D* as input file
       
    %(config)s

If the name of the output file is not specified, the script will just
add ``.uh3d`` as the extension to the input (deleting ``.tri`` if
possible).

:Versions:
    * 2015-04-17 ``@ddalle``: Version 1.0
    * 2017-04-06 ``@ddalle``: Version 1.1: JSON and MIXSUR config files
""" % _help

HELP_TRI2SURF = r"""
``cape-tri2surf``: Convert surf triangulation to AFLR3 format
==================================================================

Convert a ``.tri`` file to a AFLR3 surface format.

:Usage:
    .. code-block:: console

        $ cape-tri2surf TRI [OPTIONS]
        $ cape-tri2surf TRI SURF [OPTIONS]
        $ cape-tri2surf -i TRI [-o SURF] [OPTIONS]

:Inputs:
    * *TRI*: Input triangulation file; any readable format
    * *SURF*: Name of output ``.surf`` file
    
:Options:
    %(help)s
        
    -i TRI
        Use *TRI* as input file
        
    -o SURF
        Use *SURF* as name of created output file
       
    %(config)s
        
    --bc MAPBC
        Use *MAPBC* as boundary condition map file
    
If the name of the output file is not specified, it will just add
``.surf`` as the extension to the input (deleting ``.tri`` if possible).

:Versions:
    * 2014-06-12 ``@ddalle``: Version 1.0
    * 2015-10-09 ``@ddalle``: Version 1.1; tols and Config.xml
    * 2021-10-12 ``@ddalle``: Version 2.0; move to :mod:`tricli`
""" % _help

HELP_TRI2PLT = r"""
``cape-tri2plt``: Convert Triangulation to Tecplot PLT Format
==================================================================

Convert a Cart3D triangulation ``.tri`` or ``.triq`` file to a Tecplot
PLT file.  Each component of the triangulation is written as a separate
zone.

:Usage:
    .. code-block:: console
    
        $ cape-tri2plt TRI [PLT] [OPTIONS]
        $ cape-tri2plt -i TRI [-o PLT] OPTIONS

:Inputs:
    * *TRI*: Name of input ``.tri`` file
    * *PLT*: Name of output Tecplot ``.plt`` or ``.dat`` file

:Options:
    %(help)s
        
    -v
        Verbose output while creating PLT interface

    -i TRI
        Use *TRI* as name of created output file

    -o PLT
        Use *PLT* as input file; default is to replace ``tri``
        extension of *TRI* with ``plt``

    --dat
        Explicitly write output as ASCII Tecplot file
    
    --plt
        Explicitly use binary ``.plt`` format for output

    %(config)s

:Versions:
    * 2014-04-05 ``@ddalle``: Version 1.0
    * 2021-10-01 ``@ddalle``: Version 2.0
""" % _help


def tri2plt(*a, **kw):
    r"""Convert a UH3D triangulation file to Cart3D ``.tri`` format
    
    :Call:
        >>> tri2plt(ftri, **kw)
        >>> tri2plt(ftri, fplt, **kw)
        >>> tri2plt(i=ftri, o=fplt, **kw)
    :Inputs:
        *ftri*: :class:`str`
            Name of input file; can be any readable TRI or TRIQ format
        *fplt*: {``None``} | :class:`str`
            Name of PLT file to create; defaults to *tri* with the
            ``.tri`` replaced by ``.plt``
        *dat*: {``None``} | ``True`` | ``False``
            Write output file as ASCII format
        *plt*: {``None``} | ``true`` | ``False``
            Opposite of *dat*; default is to guess bases on *fplt*
        *c*: :class:`str`
            Surface config file, guess type from file name 
        *json*: {``None``} | :class:`str`
            JSON surface config file 
        *mixsur*: {``None``} | :class:`str`
            MIXSUR/USURP surface config file 
        *xml*: {``None``} | :class:`str`
            XML surface config file
        *v*: ``True`` | {``False``}
            Verbose output while creating PLT instance
    :Versions:
        * 2016-04-05 ``@ddalle``: Version 1.0
        * 2021-10-01 ``@ddalle``: Version 2.0
    """
    # Check for ASCII output option
    qdat = kw.get("dat")
    qplt = kw.get("plt")
    # Get input file name
    ftri = _get_i(*a, **kw)
    # Get output file name
    if qdat:
        # Default to ".dat" extension
        fplt = _get_o(ftri, "tri", "dat", *a, **kw)
    else:
        # Default to ".plt" extension
        fplt = _get_o(ftri, "tri", "plt", *a, **kw)
    # Check file name for default output type
    if qdat is not None:
        # Explicit
        qdat = qdat
    elif qplt is not None:
        # Explicit based on opposite variable
        qdat = not qplt
    else:
        # Check file name
        qdat = fplt.endswith(".dat")
    # Read TRI file
    tri = Tri(ftri)
    # Read Config file
    _read_triconfig(tri, *a, **kw)
    # Create PLT interface
    plt = Plt(triq=tri, **kw)
    # Output
    if qdat:
        # Write ASCII Tecplot DAT file
        plt.WriteDat(fplt)
    else:
        # Write PLT file
        plt.Write(fplt)


def tri2surf(*a, **kw):
    r"""Convert a triangulated surface to AFLR3 ``.surf`` format
    
    :Call:
        >>> Tri2Surf(tri, surf, bc=None)
        >>> Tri2Surf(i=tri, o=surf, bc=None)
    :Inputs:
        *tri*: :class:`str`
            Name of input file
        *surf*: :class:`str`
            Name of output file (defaults to *tri* with ``.surf`` as
            the extension in the place of ``.tri``)
        *bc*: :class:`str`
            (Optional) name of boundary condition file to apply
    :Versions:
        * 2015-11-19 ``@ddalle``: Version 1.0; :func:`Tri2Surf`
        * 2021-10-12 ``@ddalle``: Version 2.0;
            - :func:`tri2surf`
            - in :mod:`cape.tricli`
            - support all three surf config formats
    """
    # Get input file name
    ftri = _get_i(*a, **kw)
    # Get output file name
    fsurf = _get_o(ftri, "tri", "surf", *a, **kw)
    # Read TRI file
    tri = Tri(ftri)
    # Read Config file
    _read_triconfig(tri, *a, **kw)
    # Configuration
    fbc = kw.get('bc')
    # Apply configuration if requested
    if fbc:
        # Map the boundary conditions
        tri.ReadBCs_AFLR3(fbc)
    else:
        # Use defaults.
        tri.MapBCs_AFLR3()
    # Write converted file
    tri.WriteSurf(fsurf)


def tri2uh3d(*a, **kw):
    r"""Convert a UH3D triangulation file to Cart3D tri format
    
    :Call:
        >>> tri2uh3d(ftri, **kw)
        >>> tri2uh3d(ftri, fuh3d, **kw)
        >>> tri2uh3d(i=ftri, o=fuh3d, **kw)
    :Inputs:
        *ftri*: :class:`str`
            Name of input file
        *fuh3d*: :class:`str`
            Name of output file
        *c*: :class:`str`
            Surface config file, guess type from file name 
        *json*: {``None``} | :class:`str`
            JSON surface config file 
        *mixsur*: {``None``} | :class:`str`
            MIXSUR/USURP surface config file 
        *xml*: {``None``} | :class:`str`
            XML surface config file
        *h*: ``True`` | {``False``}
            Display help and exit if ``True``
    :Versions:
        * 2015-04-17 ``@ddalle``: Version 1.0
        * 2021-10-01 ``@ddalle``: Version 2.0
    """
    # Get input file name
    ftri = _get_i(*a, **kw)
    # Get output file name
    fuh3d = _get_o(ftri, "tri", "uh3d", *a, **kw)
    # Read TRI file
    tri = Tri(ftri)
    # Read Config file
    _read_triconfig(tri, *a, **kw)
    # Write the UH3D file
    tri.WriteUH3D(fuh3d)


def uh3d2tri(*a, **kw):
    r"""Convert a UH3D triangulation file to Cart3D ``.tri`` format
    
    :Call:
        >>> uh3d2tri(uh3d, tri, c=None, **kw)
        >>> uh3d2tri(i=uh3d, o=tri, c=None, **kw)
    :Inputs:
        *uh3d*: :class:`str`
            Name of input file
        *tri*: :class:`str`
            Name of output file (defaults to value of *uh3d* but with
            ``.tri`` as the extension in the place of ``.uh3d``)
        *c*: :class:`str`
            (Optional) name of configuration file to apply
        *ascii*: {``True``} | ``False``
            Write *tri* as an ASCII file (default)
        *binary*: ``True`` | {``False``}
            Write *tri* as an unformatted Fortran binary file
        *byteorder*: {``None``} | ``"big"`` | ``"little"``
            Override system byte order using either 'big' or 'little'
        *bytecount*: {``4``} | ``8``
            Use a *PREC* of 4 for single- or 8 for double-precision 
        *xtol*: {``None``} | :class:`float`
            Tolerance for *x*-coordinates to be truncated to zero
        *ytol*: {``None``} | :class:`float`
            Tolerance for *y*-coordinates to be truncated to zero
        *ztol*: {``None``} | :class:`float`
            Tolerance for *z*-coordinates to be truncated to zero
        *dx*: {``None``} | :class:`float`
            Distance to translate all nodes in *x* direction
        *dy*: {``None``} | :class:`float`
            Distance to translate all nodes in *y* direction
        *dz*: {``None``} | :class:`float`
            Distance to translate all nodes in *z* direction
    :Versions:
        * 2014-06-12 ``@ddalle``: Version 1.0
        * 2015-10-09 ``@ddalle``: Version 1.1; ``Config.xml`` and *ytol*
        * 2016-08-18 ``@ddalle``: Version 1.2; Binary output option
    """
    # Get the input file name
    fuh3d = _get_i(*a, **kw)
    # Read in the UH3D file.
    tri = Tri(uh3d=fuh3d)
    # Get file extension
    ext = tri.GetOutputFileType(**kw)
    # Default file name
    if ext == 'ascii':
        # ASCII file: use ".tri"
        ftri = _get_o(fuh3d, "uh3d", "tri", *a, **kw)
    else:
        # Binary file: use ".i.tri"
        ftri = _got_o(fuh3d, "uh3d", "i.tri", *a, **kw)
    # Read configuration if possible
    cfg = _read_config(*a, **kw)
    # Apply configuration if requested
    if cfg is not None:
        tri.config = cfg
    # Check for tolerances
    xtol = kw.get('xtol')
    ytol = kw.get('ytol')
    ztol = kw.get('ztol')
    # Apply tolerances
    if xtol is not None:
        tri.Nodes[abs(tri.Nodes[:,0])<=float(xtol), 0] = 0.0
    if ytol is not None:
        tri.Nodes[abs(tri.Nodes[:,1])<=float(ytol), 1] = 0.0
    if ztol is not None:
        tri.Nodes[abs(tri.Nodes[:,2])<=float(ztol), 2] = 0.0
    # Check for nudges
    dx = kw.get('dx')
    dy = kw.get('dy')
    dz = kw.get('dz')
    # Apply nudges
    if dx is not None:
        tri.Nodes[:,0] += float(dx)
    if dy is not None:
        tri.Nodes[:,1] += float(dy)
    if dz is not None:
        tri.Nodes[:,2] += float(dz)
    # Get write options
    tri.Write(ftri, **kw)
    

# CLI functions
def main_uh3d2tri():
    r"""CLI for :func:`uh3d2tri`

    :Call:
        >>> main_uh3d2tri()
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    _main(uh3d2tri, HELP_UH3D2TRI)


def main_tri2plt():
    r"""CLI for :func:`tri2plt`

    :Call:
        >>> main_tri2plt()
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    _main(tri2plt, HELP_TRI2PLT)


def main_tri2surf():
    r"""CLI for :func:`tri2surf`

    :Call:
        >>> main_tri2surf()
    :Versions:
        * 2021-10-12 ``@ddalle``: Version 1.0
    """
    _main(tri2surf, HELP_TRI2SURF)


def main_tri2uh3d():
    r"""CLI for :func:`tri2uh3d`

    :Call:
        >>> main_tri2uh3d()
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    _main(tri2uh3d, HELP_TRI2UH3D)


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
        

def _read_config(*a, **kw):
    r"""Read best-guess surface config file

    :Call:
        >>> cfg = _read_config(*a, **kw)
    :Inputs:
        *c*: {``None``} | :class:`str`
            Config file, type determined from file name
        *json*: {``None``} | :class:`str`
            JSON config file name
        *mixsur*: {``None``} | :class:`str`
            ``mixsur``\ /``usurp`` config file name
        *xml*: {``None``} | :class:`str`
            XML config file name
    :Outputs:
        *cfg*: :class:`ConfigXML` or similar
            Configuration instance
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Configuration
    fcfg = kw.get('c')
    fxml = kw.get("xml")
    fjson = kw.get("json")
    fmxsr = kw.get("mixsur")
    # Check options for best config format
    if fxml:
        # Directly-specified XML config
        return ConfigXML(fxml)
    if fjson:
        # Directly-specified JSON config
        return ConfigJSON(fjson)
    if fmxsr:
        # Directly-specified MIXSUR config
        return ConfigMIXSUR(fmxsr)
    # Check options for format guessed from file name
    if fcfg:
        # Guess type based on extension
        if fcfg.endswith("json"):
            # Probably a JSON config
            return ConfigJSON(fcfg)
        elif fcfg.startswith("mixsur") or fcfg.endswith(".i"):
            # Likely a MIXSUR/OVERINT input file
            return ConfigMIXSUR(fcfg)
        else:
            # Default to XML
            return ConfigXML(fcfg)
    # Check for some defaults
    if os.path.isfile("Config.xml"):
        # Use that
        return ConfigXML("Config.xml")


def _read_triconfig(tri, *a, **kw):
    r"""Read surface config file into triangulation

    :Call:
        >>> _read_triconfig(tri, *a, **kw)
    :Inputs:
        *c*: {``None``} | :class:`str`
            Config file, type determined from file name
        *tri*: :class:`Tri`
            Triangulation instance
        *json*: {``None``} | :class:`str`
            JSON config file name
        *mixsur*: {``None``} | :class:`str`
            ``mixsur``\ /``usurp`` config file name
        *xml*: {``None``} | :class:`str`
            XML config file name
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Configuration
    fcfg = kw.get('c')
    fxml = kw.get("xml")
    fjson = kw.get("json")
    fmxsr = kw.get("mixsur")
    # Check options for best config format
    if fxml:
        # Directly-specified XML config
        tri.ReadConfigXML(fxml)
    if fjson:
        # Directly-specified JSON config
        tri.ReadConfigJSON(fjson)
    if fmxsr:
        # Directly-specified MIXSUR config
        tri.ReadConfigMIXSUR(fmxsr)
    # Check options for format guessed from file name
    if fcfg:
        # Guess type based on extension
        if fcfg.endswith("json"):
            # Probably a JSON config
            tri.ReadConfigJSON(fcfg)
        elif fcfg.startswith("mixsur") or fcfg.endswith(".i"):
            # Likely a MIXSUR/OVERINT input file
            tri.ReadConfigMIXSUR(fcfg)
        else:
            # Default to XML
            tri.ReadConfigXML(fcfg)
    # Check for some defaults
    if os.path.isfile("Config.xml"):
        # Use that
        tri.ReadConfigXML("Config.xml")

