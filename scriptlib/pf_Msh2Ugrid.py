#!/usr/bin/env python
"""
Convert UH3D triangulation to Cart3D format: :mod:`pc_UH3D2Tri`
===============================================================

Convert a '.uh3d' file to a Cart3D triangulation format.

:Call:

    .. code-block:: console
    
        $ pc_UH3D2Tri.py $uh3d
        $ pc_UH3D2Tri.py -i $uh3d
        $ pc_UH3D2Tri.py $uh3d $tri
        $ pc_UH3D2Tri.py -i $uh3d -o $tri
        $ pc_UH3D2Tri.py -h

:Inputs:
    * *uh3d*: Name of input '.uh3d' file
    * *tri*: Name of output '.tri' file
    
:Options:
    -h, --help
        Display this help message and exit
        
    -i UH3D
        Use *UH3D* as input file
        
    -o TRI
        Use *TRI* as name of created output file
       
    -c XML
        Use file *XML* to map component ID numbers
        
    -xtol XTOL
        Truncate nodal coordinates within *XTOL* of x=0 plane to zero
        
    -ytol YTOL
        Truncate nodal coordinates within *YTOL* of y=0 plane to zero
        
    -ztol ZTOL
        Truncate nodal coordinates within *ZTOL* of z=0 plane to zero
    
If the name of the output file is not specified, it will just add '.tri' as the
extension to the input (deleting '.uh3d' if possible).

:Versions:
    * 2014-06-12 ``@ddalle``: First version
    * 2015-10-09 ``@ddalle``: Added tolerances and Config.xml processing
"""

# Get the CAPE mesh module.
import cape.mesh
# Module to handle inputs and os interface
import sys
# Command-line input parser
import pyCart.argread as argr

# Main function
def Msh2Ugrid(*a, **kw):
    """
    Convert a UH3D triangulation file to Cart3D tri format
    
    :Call:
        >>> Msh2Ugrid(msh, ugrid)
        >>> Msh2Ugrid(i=msh, o=ugrid)
    :Inputs:
        *msh*: :class:`str`
            Name of input file
        *ugrid*: :class:`str`
            Name of output file (defaults to value of uh3d but with ``.ugrid``
            as the extension in the place of ``.msh``
        *xtol*: :class:`float` | :class:`str`
            Tolerance for *x*-coordinates to be truncated to zero
        *ytol*: :class:`float` | :class:`str`
            Tolerance for *y*-coordinates to be truncated to zero
        *ztol*: :class:`float` | :class:`str`
            Tolerance for *z*-coordinates to be truncated to zero
    :Versions:
        * 2014-06-12 ``@ddalle``: First documented version
        * 2015-10-09 ``@ddalle``: Added ``Config.xml`` and *ytol*
    """
    # Get the file pyCart settings file name.
    if len(a) == 0:
        # Defaults
        fmsh = None
    else:
        # Use the first general input.
        fmsh = a[0]
    # Prioritize a "-i" input.
    fmsh = kw.get('i', fmsh)
    # Must have a file name.
    if fmsh is None:
        # Required input.
        print(__doc__)
        raise IOError("At least one input required.")
        sys.exit(1)
    
    # Get the file pyCart settings file name.
    if len(a) <= 2:
        # Defaults
        fugrd = fmsh.rstrip('msh') + 'ugrid'
    else:
        # Use the first general input.
        fugrd = a[1]
    # Prioritize a "-i" input.
    fugrd = kw.get('o', fugrd)
        
    # Read in the UH3D file.
    M = cape.mesh.Mesh(fmsh)
    # Process cell-to-node information
    M.GetCells()
    
    # Check for tolerances
    xtol = kw.get('xtol')
    ytol = kw.get('ytol')
    ztol = kw.get('ztol')
    # Apply tolerances
    if xtol is not None:
        M.Nodes[abs(M.Nodes[:,0])<=float(xtol), 0] = 0.0
    if ytol is not None:
        M.Nodes[abs(M.Nodes[:,1])<=float(ytol), 1] = 0.0
    if ztol is not None:
        M.Nodes[abs(M.Nodes[:,2])<=float(ztol), 2] = 0.0
    
    # Write it.
    M.WriteAFLR3ASCII(fugrd)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function.
    Msh2Ugrid(*a, **kw)
    
