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
        
    -o CRV
        Use *CRV* as name of created output file
        
    --atol ATOL
        Maximum angle between node vector and curve tangent [60]
        
    --dtol DTOL
        Max distance from node to curve as a fraction of edge length [0.05]
        
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
    fcrv = kw.get('o',   '%s.crv'   % fpre)
    # Read input files
    print("Reading TRI file: '%s" % ftri)
    tri = cape.tri.Tri(ftri)
    print("Reading STEP file: '%s" % fstp)
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
    print("Sampling curves...")
    for i in range(stp.ncrv):
        # Sample the curve
        Yi = stp.SampleCurve(i, **kw_s)
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
    print("Writing curves: '%s" % fcrv)
    stp.WritePlot3DCurves(fcrv)
    
    

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

