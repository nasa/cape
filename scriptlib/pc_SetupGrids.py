#!/usr/bin/python
"""
pc_SetupGrids
=============

Setup grids based on existing pyCart settings using defaults where necessary.

:Call:
    $ pc_SetupGrids.py
    $ pc_SetupGrids.py $json
    
:Inputs:
    *json*: Name of pyCart control file (defaults to "pyCart.json")
"""

# Get the pyCart module.
import pyCart
# Module to handle inputs and os interface
import sys
# Command-line input parser
import pyCart.argread as argr

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
else:
    # All default options
    a = []
    kw = {}
    
# Check for a help option
if kw.get('h',False) or kw.get('help',False):
    print __doc__
    sys.exit()

# Get the file pyCart settings file name.
if len(a) == 0:
    # Default file name.
    fname = 'pyCart.json'
else:
    # Use the first general input.
    fname = a[0]
    
# Read in the settings file.
cntl = pyCart.Cntl(fname)

# Create the grid folder and make the mesh.
cntl.CreateMesh()

