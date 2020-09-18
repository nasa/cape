#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Convert Cart3D TRI file to AFLR3 Surf format: ``pc_Tri2Surf.py``
==================================================================

Convert a ``.tri`` file to a AFLR3 surface format.

:Usage:
    .. code-block:: console
    
        $ pc_Tri2Surf.py TRI
        $ pc_Tri2Surf.py -i TRI
        $ pc_Tri2Surf.py UH3D SURF
        $ pc_Tri2Surf.py -i TRI -o SURF -c XML
        $ pc_Tri2Surf.py -h

:Inputs:
    * *TRI*: Name of input ``.tri`` file
    * *SURF*: Name of output ``.surf`` file
    * *XML*: Name of ``Config.xml`` file
    
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
    
If the name of the output file is not specified, it will just add
``.surf`` as the extension to the input (deleting ``.tri`` if possible).

:Versions:
    * 2014-06-12 ``@ddalle``: Version 1.0
    * 2015-10-09 ``@ddalle``: Version 1.1: tols and Config.xml
"""

# Standard library
import sys

# CAPE modules
import cape.argread
import cape.pycart


# Main function
def Tri2Surf(*a, **kw):
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
        * 2015-11-19 ``@ddalle``: Version 1.0
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
        print(__doc__)
        raise IOError("At least one input required.")
    
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
    (a, kw) = cape.argread.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        import cape.text
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Run the main function.
    Tri2Surf(*a, **kw)

