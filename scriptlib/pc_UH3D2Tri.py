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

