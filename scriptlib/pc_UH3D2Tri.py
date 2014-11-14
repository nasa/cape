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
    
If the name of the output file is not specified, it will just add '.tri' as the
extension to the input (deleting '.uh3d' if possible).

:Versions:
    * 2014-06-12 ``@ddalle``: First version
"""

# Get the pyCart module.
import pyCart
# Module to handle inputs and os interface
import sys
# Command-line input parser
import pyCart.argread as argr

# Main function
def UH3D2Tri(*a, **kw):
    """
    Convert a UH3D triangulation file to Cart3D tri format
    
    :Call:
        >>> UH3D2Tri(uh3d, tri, h=False)
        >>> UH3D2Tri(i=uh3d, o=tri, h=False)
    :Inputs:
        *uh3d*: :class:`str`
            Name of input file
        *tri*: :class:`str`
            Name of output file (defaults to value of uh3d but with ``.tri`` as
            the extension in the place of ``.uh3d``
        *h*: :class:`bool`
            Display help and exit if ``True``
    :Versions:
        * 2014-06-12 ``@ddalle``: First documented version
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
        sys.exit(1)
    
    # Get the file pyCart settings file name.
    if len(a) <= 2:
        # Defaults
        ftri = fuh3d.rstrip('.uh3d') + '.tri'
    else:
        # Use the first general input.
        ftri = a[1]
    # Prioritize a "-i" input.
    ftri = kw.get('o', ftri)
        
    # Read in the UH3D file.
    tri = pyCart.Tri(uh3d=fuh3d)
    
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
    
