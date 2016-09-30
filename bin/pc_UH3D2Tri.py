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
        
    -ascii
        Write *TRI* as an ASCII file (default)
        
    -binary, -bin
        Write *TRI* as an unformatted Fortran binary file
        
    -byteorder BO, -endian BO
        Override system byte order using either 'big' or 'little'
        
    -bytecount PREC
        Use a *PREC* of 4 for single-precision or 8 for double-precision 
        
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

# Get the pyCart module.
import pyCart
# Module to handle inputs and os interface
import sys
# Command-line input parser
import pyCart.argread as argr


import numpy as np

# Main function
def UH3D2Tri(*a, **kw):
    """
    Convert a UH3D triangulation file to Cart3D tri format
    
    :Call:
        >>> UH3D2Tri(uh3d, tri, c=None)
        >>> UH3D2Tri(i=uh3d, o=tri, c=None)
    :Inputs:
        *uh3d*: :class:`str`
            Name of input file
        *tri*: :class:`str`
            Name of output file (defaults to value of uh3d but with ``.tri`` as
            the extension in the place of ``.uh3d``
        *c*: :class:`str`
            (Optional) name of configuration file to apply
        *ascii*: {``True``} | ``False``
            Write *tri* as an ASCII file (default)
        *binary*: ``True`` | {``False``}
            Write *tri* as an unformatted Fortran binary file
        *byteorder*: {``None``} | ``"big"`` | ``"little"``
            Override system byte order using either 'big' or 'little'
        *bytecount*: {``4``} | ``8``
            Use a *PREC* of 4 for single-precision or 8 for double-precision 
        *xtol*: :class:`float` | :class:`str`
            Tolerance for *x*-coordinates to be truncated to zero
        *ytol*: :class:`float` | :class:`str`
            Tolerance for *y*-coordinates to be truncated to zero
        *ztol*: :class:`float` | :class:`str`
            Tolerance for *z*-coordinates to be truncated to zero
    :Versions:
        * 2014-06-12 ``@ddalle``: First documented version
        * 2015-10-09 ``@ddalle``: Added ``Config.xml`` and *ytol*
        * 2016-08-18 ``@ddalle``: Added binary output
    """
    # Get the file pyCart settings file name.
    if len(a) == 0:
        # Defaults
        fuh3d = None
    else:
        # Use the first general input.
        fuh3d = a[0]
    # Prioritize a "-i" input.
    fuh3d = kw.get('i', fuh3d)
    # Must have a file name.
    if fuh3d is None:
        # Required input.
        print __doc__
        raise IOError("At least one input required.")
        
    # Get binary option
    qbin = kw.get('binary', kw.get('bin', False))
    # Check for -ascii override
    if kw.get('ascii') == True:
        qbin = False
        
    # Get the file pyCart settings file name.
    if len(a) <= 2:
        # Defaults
        if qbin:
            # Binary file: use ".i.tri"
            ftri = fuh3d.rstrip('uh3d') + 'i.tri'
        else:
            # ASCII file: use ".tri"
            ftri = fuh3d.rstrip('uh3d') + 'tri'
    else:
        # Use the first general input.
        ftri = a[1]
    # Prioritize a "-i" input.
    ftri = kw.get('o', ftri)
        
    # Read in the UH3D file.
    tri = pyCart.Tri(uh3d=fuh3d)
    
    # Configuration
    fxml = kw.get('c')
    # Apply configuration if requested.
    if fxml:
        # Read the configuration.
        cfg = pyCart.config.Config(fxml)
        # Apply it.
        tri.ApplyConfig(cfg)
    
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
    if dx is not None: tri.Nodes[:,0] += float(dx)
    if dy is not None: tri.Nodes[:,1] += float(dy)
    if dz is not None: tri.Nodes[:,2] += float(dz)
    
    # Get write options
    if qbin:
        # Write binary version
        tri.WriteTriBin(ftri, 
            byteorder=kw.get('byteorder', kw.get('endian')),
            bytecount=kw.get('bytecount', 4))
    else:
        # Write it.
        tri.Write(ftri)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function.
    UH3D2Tri(*a, **kw)
    
