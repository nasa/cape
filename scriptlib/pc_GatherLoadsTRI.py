#!/usr/bin/python
"""
pc_UH3D2Tri
===========

Convert a '*.uh3d' file to a Cart3D triangulation format.

:Call:
    $ pc_UH3D2Tri.py $uh3d
    $ pc_UH3D2Tri.py -i $uh3d
    $ pc_UH3D2Tri.py -i $uh3d -o $tri

:Outputs:
    *uh3d*: Name of input '.uh3d' file
    *tri*: Name of output '.tri' file
    
If the name of the output file is not specified, it will just add '.tri' as the
extension to the input (deleting '.uh3d' if possible).
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
cart3d = pyCart.Cart3d(fname)

# Read the "loadsCC.dat" files.
cart3d.GetLoadsTRI()

# Write the files.
cart3d.WriteLoadsTRI()

