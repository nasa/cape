#!/usr/bin/python
"""
Combine triangulations: :mod:`pc_CombineTri`
============================================

Combine ".tri" files without changing component numbers.

:Call:

    .. code-block:: console
    
        $ pc_CombineTri.py $tri1 $tri2 ... $trin
        $ pc_CombineTri.py $tri1 $tri2 -o $ftri
        $ pc_CombineTri.py $tri1 $tri2 -r
        $ pc_CombineTri.py $tri1 $tri2 --raw
        $ pc_CombineTri.py -h

:Inputs:
    * *tri1*: Name of first Cart3D triangulation file
    * *tri2*: Name of second Cart3D triangulation file
    * *trin*: Name of triangulation *n*
    * *ftri*: Name of output triangulation (defautls to "Components.tri")
    
:Options:
    * *h*: Display this help and exit
    * *o*: Specify an output file name other than "Components.tri"
    * *r*, *raw*: Leave component indices unchanged when appending
    
If the name of the output file is not specified, it will just add '.tri' as the
extension to the input (deleting '.uh3d' if possible).
"""

# Get the pyCart module.
import pyCart
# Module to handle inputs and os interface
import sys
# Command-line input parser
import cape.argread as argr

# Main function
def CombineTri(*a, **kw):
    """
    Combine :file:`.tri` files without changing component numbers
    
    :Call:
        
        >>> CombineTri(tri1, tri2, ..., o="Components.tri", r=False)
        
    :Inputs:
        *tri1*: :class:`str`
            Name of first input Cart3D triangulation file
        *tri2*: :class:`str`
            Name of second input Cart3D triangulation file
        *o*: :class:`str`
            Name of output triangulation file
        *r*: :class:`bool`
            Component IDs are not altered if ``True``
    """
    # Versions:
    #  2014.06.12 @ddalle  : First version
    
    # Check for trivial combinations.
    if len(a) < 2:
        # Required input.
        print(__doc__)
        raise IOError("At least two inputs required.")
        sys.exit(1)
    
    # Get the output file name.
    ftri = kw.get("o", "Components.tri")
    # Check for raw status.
    qraw = kw.get("r", False) or kw.get("raw", False)
    
    # Read the first triangulation.
    tri = pyCart.Tri(a[0])
    # Loop through the remaining triangulations.
    for f in a[1:]:
        # Read the triangulation
        t = pyCart.Tri(f)
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
    
