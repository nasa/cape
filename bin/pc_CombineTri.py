#!/usr/bin/python
# -*- coding: utf-8 -*-
r"""
Combine triangulations: ``pc_CombineTri.py``
============================================

Combine ``.tri`` files without changing component numbers.

:Usage:
    .. code-block:: console
    
        $ pc_CombineTri.py TRI1 TRI2 [...] [OPTIONS]
        $ pc_CombineTri.py -h

:Inputs:
    * *TRI1*: Name of first Cart3D triangulation file
    * *TRI2*: Name of second Cart3D triangulation file
    * *TRIn*: Name of triangulation *n*
    
:Options:
    -h, --help
        Display this help and exit

    -o OTRI
        Name of output file (default: ``"Components.tri"``)

    -r, --raw
        Leave component indices unchanged when appending

:Versions:
    * 2014-10-06 ``@ddalle``: Version 1.0
"""

# Standard library modules
import sys

# Local modules
import cape.pycart
import cape.argread as argr


# Main function
def CombineTri(*a, **kw):
    r"""Combine ``.tri`` files without changing component numbers
    
    :Call:
        >>> CombineTri(tri1, tri2, *a, o="Components.tri", r=False)
    :Inputs:
        *tri1*: :class:`str`
            Name of first input Cart3D triangulation file
        *tri2*: :class:`str`
            Name of second input Cart3D triangulation file
        *o*: :class:`str`
            Name of output triangulation file
        *r*: ``True`` | {``False``}
            Component IDs are not altered if ``True``
    :Versions:
        * 2014-06-12: ``@ddalle``: Version 1.0
    """
    # Check for trivial combinations.
    if len(a) < 2:
        # Required input.
        raise IOError("At least two inputs required.")
    
    # Get the output file name.
    ftri = kw.get("o", "Components.tri")
    # Check for raw status.
    qraw = kw.get("r", False) or kw.get("raw", False)
    
    # Read the first triangulation.
    tri = cape.pycart.Tri(a[0])
    # Loop through the remaining triangulations.
    for f in a[1:]:
        # Read the triangulation
        t = cape.pycart.Tri(f)
        # Add each triangulation subsequently.
        if qraw:
            # Append without changing CompIDs at all.
            tri.AddRawCompID(t)
        else:
            # Add but increase the CompIDs of the second tri.
            tri.Add(t)
    
    # Status update
    print("Writing '%s' with %i tris and %i components" % 
        (ftri, tri.nTri, max(tri.CompID)))
    # Write it.
    tri.Write(ftri)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        import cape.text
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Run the main function.
    CombineTri(*a, **kw)
    
