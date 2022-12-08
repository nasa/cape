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
from .step import STEP
from .tri import Tri, Triq


# Template help messages
_help_help = r"""-h, --help
        Display this help message and exit"""
_help_sample = r"""
    -n N
        Use a minimum of *N* segments per curve (defaults to ``3``)

    --ds DS
        Use a maximum arc length of *DS* on each curve

    --dth THETA
        Make sure maximum turning angle is below *THETA*

    --da PHI
        Make sure maximum turning angle times length of adjoining
        segments is less than or equal to *PHI*"""
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
    "sample": _help_sample,
}

# Lists of file extensions
STEP_EXTS = [
    "STEP",
    "STP",
    "step",
    "stp"
]
TRI_EXTS = [
    "lb4.tri",
    "b4.tri",
    "i.tri",
    "lr4.tri",
    "r4.tri",
    "tri",
    "lb4.triq",
    "b4.triq",
    "i.triq",
    "lr4.triq",
    "r4.triq",
    "triq",
]

# Help messages
HELP_STEP2CRV = r"""
``cape-step2crv``: Convert STEP file to Plot3D multiple curve file
===================================================================

Create a Plot3D discretized curve file from a STEP file using various
maximum spacing command-line options.

:Usage:
    .. code-block:: console
    
        $ cape-step2crv STP [OPTIONS]
        $ cape-step2crv STP CRV [OPTIONS]
        $ cape-step2crv -i STP [-o CRV] [OPTIONS]

:Inputs:
    * *STP*: Name of input ``'.stp`` or ``.step`` file
    * *CRV*: Name of output Plot3D file
    
:Options:
    %(help)s
        
    -i STP
        Use *STP* as input file
        
    -o CRV
        Use *CRV* as name of created output file
        
    %(sample)s
    
    --link
        Link curves and sort by ascending *x* coordinate (default)
        
    --link AXIS
        Link curves and sort by *AXIS*, for example ``+x``, ``-y``, etc.
        
    --no-link
        Do not link curves together or sort
        
    --xtol XTOL
        Truncate nodal coordinates within *XTOL* of x=0 plane to zero
        
    --ytol YTOL
        Truncate nodal coordinates within *YTOL* of y=0 plane to zero
        
    --ztol ZTOL
        Truncate nodal coordinates within *ZTOL* of z=0 plane to zero
    
:Versions:
    * 2016-05-10 ``@ddalle``: Version 1.0
    * 2021-10-15 ``@ddalle``: Version 2.0
""" % _help


HELP_STEPTRI2CRV = r"""
``cape-steptri2crv``: Extract TRI file nodes on STEP curves
===============================================================

Convert a STEP file to a Plot3D multiple curve file by sampling curves
from the STEP file at nodes from a surface triangulation. The result is
a multiple-curve Plot3D curve file

:Usage:
    .. code-block:: console
    
        $ cape-steptri2crv STP [TRI [CRV]] [OPTIONS]
        $ cape-steptri2crv --stp STP [--tri TRI] [--crv CRV] [OPTIONS]

:Inputs:
    * *STP*: name of input STEP file
    * *TRI*: name of input TRI file (or *STP* with new extension)
    * *CRV*: name of output curve file (or *STP* with new extension)
    
:Options:
    %(help)s
        
    --stp STP
        Use *STP* as input STEP file
        
    --tri TRI
        Use *TRI* as input triangulation file

    --crv CRV
        Use *CRV* as name of created output file

    --atol ATOL
        Maximum angle between node vector and curve tangent [60]

    --dtol DTOL
        Max dist. from node to curve as a fraction of edge length [0.05]

    --ascii
        Write text curves file instead of binary

    --lr8
        Write double-precision little-endian curves

    --r8
        Write double-precision big-endian curves

    --lr4
        Write single-precision little-endian curves

    --r4
        Write single-precision big-endian curves

    --endian BO
        Use non-default byte order, "big" or "little"

    --sp
        Write single-precision curve file (default is double)

    %(sample)s

:Versions:
    * 2016-09-29 ``@ddalle``: Version 1.0
""" % _help


HELP_UH3D2TRI = r"""
``cape-uh3d2tri``: Convert UH3D triangulation to Cart3D format
===============================================================

Convert a ``.uh3d`` file to a Cart3D triangulation format.
    
If the name of the output file is not specified, it will just add '.tri'
as the extension to the input (deleting '.uh3d' if possible).

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

If the name of the output file is not specified, the script will just
add ``.uh3d`` as the extension to the input (deleting ``.tri`` if
possible).

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

:Versions:
    * 2015-04-17 ``@ddalle``: Version 1.0
    * 2017-04-06 ``@ddalle``: Version 1.1: JSON and MIXSUR config files
""" % _help


HELP_TRI2SURF = r"""
``cape-tri2surf``: Convert surf triangulation to AFLR3 format
==================================================================

Convert a ``.tri`` file to a AFLR3 surface format.
    
If the name of the output file is not specified, it will just add
``.surf`` as the extension to the input (deleting ``.tri`` if possible).

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

    --triq
        Manually specify that input file is a ``.triq`` file

    %(config)s

:Versions:
    * 2014-04-05 ``@ddalle``: Version 1.0
    * 2021-10-01 ``@ddalle``: Version 2.0
""" % _help


# API functions
def step2crv(*a, **kw):
    r"""Write the curves from a STEP file to Plot3D multiple curve file
    
    :Call:
        >>> step2crv(fstp, fcrv, **kw)
        >>> step2crv(i=fstp, o=fcrv, **kw)
    :Inputs:
        *fstp*: :class:`str`
            Name of input file
        *fcrv*: :class:`str`
            Name of output file (defaults to value of *fstp* but with
            ``.crv`` in the place of ``.stp`` or ``.step``)
        *n*: :class:`int`
            Number of intervals to use
        *ds*: :class:`float`
            Upper bound of uniform spacing
        *dth*: :class:`float` | {``None``}
            Maximum allowed turning angle in degrees
        *da*: :class:`float` | {``None``}
            Maximum allowed length-weighted turning angle
        *link*: ``True`` | ``False`` | {``"x"``} | ``"-x"``
            Whether or not to link curves and if so using which axis to
            use for sorting
        *xtol*: :class:`float` | :class:`str`
            Tolerance for *x*-coordinates to be truncated to zero
        *ytol*: :class:`float` | :class:`str`
            Tolerance for *y*-coordinates to be truncated to zero
        *ztol*: :class:`float` | :class:`str`
            Tolerance for *z*-coordinates to be truncated to zero
    :Versions:
        * 2016-05-10 ``@ddalle``: Version 1.0
        * 2021-10-15 ``@ddalle``: Version 2.0; in :mod:`cape.tricli`
    """
    # Get input file name
    fstp = _get_i(*a, **kw)
    # Get output file name
    fcrv = _get_o(fstp, STEP_EXTS, "crv", *a, **kw)
    # Options
    xtol = kw.get("xtol")
    ytol = kw.get("ytol")
    ztol = kw.get("ztol")
    if isinstance(xtol, str):
        xtol = float(xtol)
    if isinstance(ytol, str):
        ytol = float(ytol)
    if isinstance(ztol, str):
        ztol = float(ztol)
    # Read in the STEP file
    stp = STEP(fstp, xtol=xtol, ytol=ytol, ztol=ztol)
    # Sampling options
    n = kw.get('n', 3)
    da = kw.get('da')
    ds = kw.get('ds')
    dth = kw.get('dth')
    tol = kw.get('tol', 1.0)
    # Convert as necessary
    if n is not None:
        n = int(n)
    if ds is not None:
        ds  = float(ds)
    if dth is not None:
        dth = float(dth)
    if da is not None:
        da = float(da)
    if tol is not None:
        tol = float(tol)
    # Sample (discretize) curves
    stp.SampleCurves(n=n, ds=ds, dth=dth, da=da)
    # Get link options
    nolink = kw.get('no-link') 
    axis = kw.get('link', 'x')
    # Link/sort as requested
    if not nolink and axis is True:
        # Default sorting
        stp.LinkCurves(ds=tol)
    elif not nolink and axis:
        # Specialized sorting
        stp.LinkCurves(axis=axis, ds=tol)
    # Write the curves
    if kw.get('ascii', False):
        # Write ASCII file
        stp.WritePlot3DCurvesASCII(fcrv)
    elif kw.get('lb8', kw.get('lr8', False)):
        # Write little-endian double
        stp.WritePlot3DCurvesBin(fcrv, endian='little', single=False)
    elif kw.get('b8', kw.get('r8', False)):
        # Write big-endian double
        stp.WritePlot3DCurvesBin(fcrv, endian='big', single=False)
    elif kw.get('lb4', kw.get('lr4', False)):
        # Write little-endian single
        stp.WritePlot3DCurvesBin(fcrv, endian='little', single=True)
    elif kw.get('b4', kw.get('r4', False)):
        # Write big-endian single
        stp.WritePlot3DCurvesBin(fcrv, endian='big', single=True)
    else:
        # Process endianness
        bo = kw.get('endian')
        # Process precision
        sp = kw.get('sp', False)
        # Write binary file
        stp.WritePlot3DCurvesBin(fcrv, endian=bo, single=sp)


def steptri2crv(*a, **kw):
    r"""Write Plot3D curves of ``.tri`` nodes near ``.stp`` curves

    Read curves from a STEP file and use these to subset nodes from a
    surface triangulation.  Each curve is written as a series of points,
    and the combined output is written to a Plot3D multiple curve file.

    :Call:
        >>> steptri2crv(fstp, **kw)
        >>> steptri2crv(fstp, ftri, **kw)
        >>> steptri2crv(fstp, ftri, fcrv, **kw)
        >>> steptri2crv(stp=fstp, tri=ftri, o=fcrv, **kw)
    :Inputs:
        *fstp*: :class:`str`
            Name of input STEP file
        *ftri*: :class:`str`
            Name of input TRI file (defaults to *fstp* with ``.tri``
            in place of ``.stp`` or ``.step``)
        *fcrv*: :class:`str`
            Name of output Plot3D curve file (defaults to *fstp* with
            ``.crv`` in place of ``.stp`` or ``.step``)
        *sp*: ``True`` | {``False``}
            Write curves as single-precision file
        *ascii*: ``True`` | {``False``}
            Write curves as text file
        *endian*: {``None``} | ``"big"`` | ``"little"``
            Byte order
        *r4*, *b4*: ``True`` | {``False``}
            Write single-precision big-endian
        *r8*, *b8*: ``True`` | {``False``}
            Write double-precision big-endian
        *lr4*, *lb4*: ``True`` | {``False``}
            Write single-precision little-endian
        *lr8*, *lb8*: ``True`` | {``False``}
            Write double-precision little-endian
    :Versions:
        * 2016-09-29 ``@ddalle``: Version 1.0
        * 2021-10-15 ``@ddalle``: Version 2.0
    """
    # Get first input file
    fstp = _get_i(*a, _key="stp", **kw)
    # Get second input file, then output file
    ftri = _get_o(fstp, STEP_EXTS, "tri", *a, _arg=1, _key="tri", **kw)
    fcrv = _get_o(fstp, STEP_EXTS, "crv", *a, _arg=2, _key="o", **kw)
    # Read input files
    print("  Reading TRI file: '%s" % ftri)
    tri = Tri(ftri)
    print("  Reading STEP file: '%s" % fstp)
    stp = STEP(fstp)
    # Get the edges of the triangles
    tri.GetEdges()
    # Initialize curves
    X = []
    # Options for initial curve sampling
    kw_s = {
        'n':   kw.get('n', 100),
        'ds':  kw.get('ds'),
        'dth': kw.get('dth'),
        'da':  kw.get('da')
    }
    # Loop through curves
    print("  Sampling curves...")
    for i in range(stp.ncrv):
        # Sample the curve
        Yi = stp.SampleCurve(i, **kw_s)
        # Get the nodes for this curve
        Xi = tri.TraceCurve(Yi, **kw)
        # Check for valid curve
        if len(Xi) > 1:
            # Valid curve
            X.append(Xi)
            continue
        # If reached here, tracing failed; try reverse curve
        Yi = np.flipud(Yi)
        # Get the nodes for this curve
        Xi = tri.TraceCurve(Yi, **kw)
        # Check for valid curve
        if len(Xi) > 1:
            # Valid curve
            X.append(Xi)
    # Trick the STEP object into using these curves
    stp.ncrv = len(X)
    stp.crvs = X
    # Write the curves
    print("  Writing curves: '%s" % fcrv)
    if kw.get('ascii', False):
        # Write ASCII file
        stp.WritePlot3DCurvesASCII(fcrv)
    elif kw.get('lb8', kw.get('lr8', False)):
        # Write little-endian double
        stp.WritePlot3DCurvesBin(fcrv, endian='little', single=False)
    elif kw.get('b8', kw.get('r8', False)):
        # Write big-endian double
        stp.WritePlot3DCurvesBin(fcrv, endian='big', single=False)
    elif kw.get('lb4', kw.get('lr4', False)):
        # Write little-endian single
        stp.WritePlot3DCurvesBin(fcrv, endian='little', single=True)
    elif kw.get('b4', kw.get('r4', False)):
        # Write big-endian single
        stp.WritePlot3DCurvesBin(fcrv, endian='big', single=True)
    else:
        # Process endianness
        bo = kw.get('endian')
        # Process precision
        sp = kw.get('sp', False)
        # Write binary file
        stp.WritePlot3DCurvesBin(fcrv, endian=bo, single=sp)


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
        *triq*: ``True`` | ``False``
            Manually specify ``triq`` file input (default determined by
            file extension of *ftri*)
    :Versions:
        * 2016-04-05 ``@ddalle``: Version 1.0
        * 2021-10-01 ``@ddalle``: Version 2.0
    """
    # Check for ASCII output option
    qdat = kw.get("dat")
    qplt = kw.get("plt")
    # Get input file name
    ftri = _get_i(*a, **kw)
    # Option for TRIQ
    qtriq = kw.get("triq", ftri.endswith("triq"))
    # Get output file name
    if qdat:
        # Default to ".dat" extension
        fplt = _get_o(ftri, TRI_EXTS, "dat", *a, **kw)
    else:
        # Default to ".plt" extension
        fplt = _get_o(ftri, TRI_EXTS, "plt", *a, **kw)
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
    if qtriq:
        # Read with state
        tri = Triq(ftri)
    else:
        # No state variables
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
    fsurf = _get_o(ftri, TRI_EXTS, "surf", *a, **kw)
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
    fuh3d = _get_o(ftri, TRI_EXTS, "uh3d", *a, **kw)
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
def main_step2crv():
    r"""CLI for :func:`step2crv`

    :Call:
        >>> main_step2crv()
    :Versions:
        * 2021-10-15 ``@ddalle``: Version 1.0
    """
    _main(step2crv, HELP_STEP2CRV)


def main_steptri2crv():
    r"""CLI for :func:`steptri2crv`

    :Call:
        >>> main_steptri2crv()
    :Versions:
        * 2021-10-15 ``@ddalle``: Version 1.0
    """
    _main(steptri2crv, HELP_STEPTRI2CRV)


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


def main_uh3d2tri():
    r"""CLI for :func:`uh3d2tri`

    :Call:
        >>> main_uh3d2tri()
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    _main(uh3d2tri, HELP_UH3D2TRI)


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
        *_arg*: {``0``} | :class:`int`
            Positional parameter to use for input
        *_key*: {``"i"``} | :class:`str`
            Name of keyword argument to use
        *_vdef*: {``None``} | :class:`str`
            Default value to use
        *a[_arg]*: :class:`str`
            Input file name specified as first positional arg
        *i*: :class:`str`
            Input file name as kwarg; supersedes *a*
    :Outputs:
        *fname_in*: :class:`str`
            Input file name
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
        * 2021-10-15 ``@ddalle``: Version 1.1
    """
    # Argument index
    n = kw.pop("_arg", 0)
    col = kw.pop("_key", "i")
    vdef = kw.pop("_vdef", None)
    # Get the input file name
    if len(a) <= n:
        # Defaults
        fname_in = vdef
    else:
        # Use the first general input
        fname_in = a[n]
    # Prioritize a "-i" input
    fname_in = kw.get(col, fname_in)
    # Must have a file name.
    if fname_in is None:
        # Required input.
        raise ValueError(
            "Required at least %i arg(s) or '%s' kwarg" % (n + 1, col))
    # Output
    return fname_in


def _get_o(fname_in, ext1, ext2, *a, **kw):
    r"""Process output file name

    :Call:
        >>> fname_out = _get_o(fname_in, ext1, ext2, *a, **kw)
        >>> fname_out = _get_o(fname_in, exts1, ext2, *a, **kw)
    :Inputs:
        *fname_in*: :class:`str`
            Input file name
        *exts1*: :class:`list`\ [:class:`str`]
            List of expected file extensions for *fname_in*
        *ext1*: :class:`str`
            Expected file extension for *fname_in*
        *ext2*: :class:`str`
            Default file extension for *fname_out*
        *_arg*: {``1``} | :class:`int`
            Positional parameter to use for input
        *_key*: {``"o"``} | :class:`str`
            Name of keyword argument to use
        *_vdef*: {``swap_ext(fname_in, ext1, ext2)``} | :class:`str`
            Default value to use
        *a[_arg]*: :class:`str`
            Input file name specified as first positional arg
        *o*: :class:`str`
            Output file name as kwarg; supersedes *a*
    :Outputs:
        *fname_out*: :class:`str`
            Output file name
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Form default
    fname_out = _swap_ext(fname_in, ext1, ext2)
    # Set default options for output
    kw.setdefault("_arg", 1)
    kw.setdefault("_key", "o")
    kw.setdefault("_vdef", fname_out)
    # Use input detector
    return _get_i(*a, **kw)


def _swap_ext(fname_in, ext1, ext2):
    r"""Strip one file extension and replace with another

    :Call:
        >>> fname_out = _swap_ext(fname_in, ext1, ext2)
        >>> fname_out = _swap_ext(fname_in, exts1, ext2)
    :Inputs:
        *fname_in*: :class:`str`
            Input file name
        *exts1*: :class:`list`\ [:class:`str`]
            List of expected file extensions for *fname_in*
        *ext1*: :class:`str`
            Expected file extension for *fname_in*
        *ext2*: :class:`str`
            Default file extension for *fname_out*
    :Outputs:
        *fname_out*: :class:`str`
            Output file name
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Get list of possible extensions
    if not isinstance(ext1, (list, tuple)):
        # Singleton list
        ext1 = [ext1]
    # Strip *ext1* as starter for default
    for extj in ext1:
        if fname_in.endswith(extj):
            # Strip expected file extension
            fname_out = fname_in[:-len(extj)]
            break
    else:
        # Use current file name since extension not found
        fname_out = fname_in + "."
    # Add supplied *ext2* extension for output
    return fname_out + ext2
        

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

