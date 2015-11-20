#!/usr/bin/env python
"""
Convert Cart3D triangulation to AFLR3 Surf format: :mod:`pc_Tri2Surf`
=====================================================================

Convert a ``.tri`` file to a AFLR3 surface format.

:Call:

    .. code-block:: console
    
        $ pc_Tri2Surf.py $tri
        $ pc_Tri2Surf.py -i $tri
        $ pc_Tri2Surf.py $uh3d $surf
        $ pc_Tri2Surf.py -i $tri -o $surf -c $xml
        $ pc_Tri2Surf.py -h

:Inputs:
    * *tri*: Name of input ``.tri`` file
    * *surf*: Name of output ``.surf`` file
    * *xml*: Name of ``Config.xml`` file
    
:Options:
    -h, --help
        Display this help message and exit
        
    -i TRI
        Use *TRI* as input file
        
    -o SURF
        Use *SURF* as name of created output file
       
    -c XML
        Use file *XML* to map component ID numbers
        
    -bc MAPBC
        Use *MAPBC* as boundary condition map file (not implemented)
    
If the name of the output file is not specified, it will just add ``.surf`` as
the extension to the input (deleting ``.tri`` if possible).

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

# Main function
def Tri2Surf(*a, **kw):
    """
    Convert a UH3D triangulation file to Cart3D tri format
    
    :Call:
        >>> Tri2Surf(tri, surf, bc=None)
        >>> Tri2Surf(i=tri, o=surf, bc=None)
    :Inputs:
        *tri*: :class:`str`
            Name of input file
        *surf*: :class:`str`
            Name of outpu file (defaults to value of tri but with ``.surf`` as
            the extension in the place of ``.tri``
        *bc*: :class:`str`
            (Optional) name of boundary condition file to apply
    :Versions:
        * 2015-11-19 ``@ddalle``: First version
    """
    # Get the file pyCart settings file name.
    if len(a) == 0:
        # Defaults
        ftri = None
    else:
        # Use the first general input.
        ftri = a[0]
    # Prioritize a "-i" input.
    ftri = kw.get('i', ftri)
    # Must have a file name.
    if ftri is None:
        # Required input.
        print __doc__
        raise IOError("At least one input required.")
        sys.exit(1)
    
    # Get the file pyCart settings file name.
    if len(a) <= 2:
        # Defaults
        fsurf = ftri.rstrip('tri') + 'surf'
    else:
        # Use the first general input.
        ftri = a[1]
    # Prioritize a "-i" input.
    fsurf = kw.get('o', fsurf)
    
    # Configuration
    fxml = kw.get('c')
        
    # Read in the UH3D file.
    if fxml:
        # Read with config.
        tri = pyCart.Tri(ftri, c=fxml)
    else:
        # Read without config
        tri = pyCart.Tri(ftri)
        
    # Configuration
    fbc = kw.get('bc')
    # Apply configuration if requested.
    if fbc:
        # Map the boundary conditions
        tri.ReadBCs_AFLR3(fbc)
    else:
        # Use defaults.
        tri.MapBCs_AFLR3()
    
    # Write it.
    tri.WriteSurf(fsurf)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function.
    Tri2Surf(*a, **kw)
    
