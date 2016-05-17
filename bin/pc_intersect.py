#!/usr/bin/env python
"""
Intersect Multiple-Component Triangulation Files
===============================================================

Each file is considered a single water-tight triangulation.  The Cart3D
``intersect`` tool, which is used to perform the triangulation intersections,
requires each closed body to have the same component ID.  This tool allows the
user to intersect bodies which have multiple components per body by wrapping
``intersect`` in a couple of other relatively simple steps.

:Call:

    .. code-block:: console
    
        $ pc_intersect.py tri1 tri2 ... [OPTIONS]

:Inputs:
    * *tri1*: Name of the first triangulation (``tri``, ``uh3d``, ...)
    * *tri2*: Name of the second triangulation
    
:Options:
    -h, --help
        Display this help message and exit
        
    -o TRI
        Use *TRI* as name of intersected tri file
    
Triangulation formats (Cart3D tri, UH3D, UNV, or AFLR3 surf) are determined
automatically from the extensions of *tri1*, *tri2*, ...  The default name for
the output file is based on *tri1* with a ``.i`` inserted into the file name,
for example ``.i.tri``.

:Versions:
    * 2016-05-17 ``@ddalle``: First version
"""

# Get the pyCart module.
from cape.tri import ReadTriFile
# CAPE binaries
import cape.bin
# Module to handle inputs and os interface
import sys
# Command-line input parser
import cape.argread as argr

# Main function
def Intersect(*a, **kw):
    """
    Convert a UH3D triangulation file to Cart3D tri format
    
    :Call:
        >>> Intersect(tri1, tri2, ..., o=None)
    :Inputs:
        *tri1*: :class:`str`
            Name of first input file
        *tri2*: :class:`str`
            Name of second input file
        *o*: :class:`str`
            Name of output file prefix, defaults to prefix of *tri1*
    :Versions:
        * 2016-05-17 ``@ddalle``: First version
    """
    # Get the file pyCart settings file name.
    if len(a) < 2:
        # Need at least two triangulations
        raise IOError("At least two triangulations required")
    # Loop through inputs
    for i in range(len(a)):
        # Get file name
        itri = a[i]
        # Get extension
        ext = itri.split('.')
        # Check for file name with no '.'
        if len(ext) > 1:
            # Get the extension
            ext = ext[-1]
        else:
            # Assume Cart3D tri
            ext = 'tri'
            
        # Read the file
        if i == 0:
            # Default prefix
            proj = itri.rstrip(ext).rstrip('.')
            # Read initial tri
            tri = ReadTriFile(itri)
            # Save number of triangles
            tri.iTri = [tri.nTri]
        else:
            # Append triangulation
            tri.Add(ReadTriFile(itri))
            # Save the number of triangles.
            tri.iTri.append(tri.nTri)
    
    # Get output prefix
    proj = kw.get('o', proj)
    
    # Write the so-called volume tri
    tri.WriteVolTri('%s.tri' % proj)
    # Write the current tri
    tri.Write('%s.c.tri' % proj)
    
    # Run intersect.
    cape.bin.intersect(i='%s.tri', o='%s.o.tri')
    # Read the original triangulation.
    tric = Tri('%s.c.tri' % proj)
    # Read the intersected triangulation.
    trii = Tri('%s.o.tri' % proj)
    # Read the pre-intersection triangulation.
    tri0 = Tri('%s.tri' % proj)
    # Map the Component IDs.
    trii.MapCompID(tric, tri0)
    # Write the triangulation.
    trii.Write('%s.i.tri' % proj)
    
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
    Intersect(*a, **kw)
    
