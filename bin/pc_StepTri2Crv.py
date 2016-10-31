#!/usr/bin/env python
"""
Conver STEP File by Sampling Nodes from a TRI File
==================================================

Convert a STEP file to a Plot3D multiple curve file by sampling curves from the
STEP file at nodes from a surface triangulation

:Call:
    
    .. code-block:: console
    
        $ pc_StepTri2Crv.py [OPTIONS]
        $ pc_StepTri2Crv.py PRE [OPTIONS]

:Inputs:
    *PRE*: Use ``$PRE.stp`` and ``$PRE.i.tri`` as input files
    
:Options:
    -h, --help
        Display this help message and exit
        
    --stp STP
        Use *STP* as input STEP file
        
    --tri TRI
        Use *TRI* as input triangulation file
        
    --crv CRV
        Use *CRV* as name of created output file
        
    --ascii
        Write text curves file instead of binary
        
    --atol ATOL
        Maximum angle between node vector and curve tangent [60]
        
    --dtol DTOL
        Max distance from node to curve as a fraction of edge length [0.05]
        
    -n N
        Use a minimum of *N* segments per curve (defaults to ``3``)
        
    -ds DS
        Use a maximum arc length of *DS* on each curve
        
    -dth THETA
        Make sure maximum turning angle is below *THETA*
        
    -da PHI
        Make sure maximum turning angle times length of adjoining segments is
        less than or equal to *PHI*
        
    --endian BO
        Use non-default byte order, "big" or "little"
        
    --sp
        Write single-precision curve file (default is double)
        
:Versions:
    * 2016-09-29 ``@ddalle``: First version
"""

# System interface
import sys
# pyCart geometry modules
import cape.step
import cape.tri
# Command-line input parser
import cape.argread as argr

# Main function
def StepTri2Crv(*a, **kw):
    """
    Read curves from a STEP file and use these to subset nodes from a surface
    triangulation.  Each curve is written as a series of points, and the
    combined output is written to a Plot3D multiple curve file
    
    :Call:
        >>> StepTri2Crv(fpre, stp=fstp, tri=ftri, o=fcrv, **kw)
    :Inputs:
        *fpre*: :class:`str`
            Set default file names as $fpre.stp, $fpre.i.tri, and $fpre.crv
        *fstp*: :class:`str`
            Name of input STEP file
        *ftri*: :class:`str`
            Name of input TRI file
        *fcrv*: :class:`str`
            Name of output file (defaults to value of *fstp* but with ``.crv``
            as the extension in the place of ``.stp`` or ``.step``)
        *sp*: ``True`` | {``False``}
            Write curves as single-precision file
        *ascii*: ``True`` | {``False``}
            Write curves as text file
        *endian*: {``None``} | ``"big"`` | ``"little"``
            Byte order
    :Versions:
        * 2016-09-29 ``@ddalle``: First version
    """
    # Get the prefix
    if len(a) == 0:
        # No common prefix
        fpre = ""
    else:
        # Common prefix
        fpre = a[0]
    # Process file names
    fstp = kw.get('stp', '%s.stp'   % fpre)
    ftri = kw.get('tri', '%s.i.tri' % fpre)
    fcrv = kw.get('crv', kw.get('o', '%s.crv' % fpre))
    # Read input files
    print("  Reading TRI file: '%s" % ftri)
    tri = cape.tri.Tri(ftri)
    print("  Reading STEP file: '%s" % fstp)
    stp = cape.step.STEP(fstp)
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
        Yi = cape.tri.np.flipud(Yi)
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
    elif kw.get('lb8', False):
        # Write little-endian double
        stp.WritePlot3DCurvesBin(fcrv, endian='little', single=False)
    elif kw.get('b8', False):
        # Write big-endian double
        stp.WritePlot3DCurvesBin(fcrv, endian='big', single=False)
    elif kw.get('lb4', False):
        # Write little-endian single
        stp.WritePlot3DCurvesBin(fcrv, endian='little', single=True)
    elif kw.get('b4', False):
        # Write big-endian single
        stp.WritePlot3DCurvesBin(fcrv, endian='big', single=True)
    else:
        # Process endianness
        bo = kw.get('endian')
        # Process precision
        sp = kw.get('sp', False)
        # Write binary file
        stp.WritePlot3DCurvesBin(fcrv, endian=bo, single=sp)
    
    

# Only process inputs if called as a script
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function.
    StepTri2Crv(*a, **kw)

