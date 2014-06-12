#!/usr/bin/python
"""
Grid setup script: :mod:`pc_SetupGrids`
=======================================

Setup grids based on existing pyCart settings using defaults where necessary.

This will create a "Grid/" folder or a list of grid folders if there are
trajectory variables that require separate meshes each time the variable
changes.  Then it will enter each folder and run "autoInputs", "cubes", and
"mgPrep" according to the global settings. 

:Call:
    .. code-block:: console
        
        $ pc_SetupGrids.py
        $ pc_SetupGrids.py $json
        $ pc_SetupGrids.py -h
    
:Inputs:
    *json*: Name of pyCart control file (defaults to "pyCart.json")

:Options:
    *h*: Display this help and exit
"""

# Get the pyCart module.
import pyCart
# Module to handle inputs and os interface
import sys
# Command-line input parser
import pyCart.argread as argr
    
# Main function
def SetupGrids(*a, **kw):
    """
    Main function for ``pc_SetupGrids.py`` script
    
    :Call:
        >>> SetupGrids(fname, h=False)
        
    :Inputs:
        *fname*: :class:`str`
            Name of global pyCart settings file, default is ``'pyCart.json'``
        *h*: :class:`bool`
            If ``True``, show help and exit
    """
    # Versions:
    #  2014.06.12 @ddalle  : Documented version
    
    # Get the file pyCart settings file name.
    if len(a) == 0:
        # Default file name.
        fname = 'pyCart.json'
    else:
        # Use the first general input.
        fname = a[0]
        
    print fname
    # Read in the settings file.
    cart3d = pyCart.Cart3d(fname)
    
    # Create the grid folder and make the mesh.
    cart3d.CreateMesh()


# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function.
    SetupGrids(*a, **kw)

