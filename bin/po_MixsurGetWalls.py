#!/usr/bin/env python
"""
Renumber MIXSUR input subsets to account for no-wall grids
==========================================================

This script generates a new MIXSUR input file with renumbered grids such that
any grids without walls are eliminated from the count.

:Usage:
    
    .. code-block:: bash
    
        $ po_MixsurGetWalls.py [FMIXSUR FSPLITMX OMIXSUR] [OPTIONS]
        
:Inputs:
    
    *FMIXSUR*: Name of original *mixsur* input file {``mixsur.i``}
    *FSPLITMX*: Name of *splitmx*/*splitmq* input file {``splitmx.i``}
    *OMIXSUR*: Name of modified *mixsur* file {``mixsur.surf.i``}
    
:Options:
    
    -h, --help
        Display this help message and quit
        
    -i FMIXSUR
        Use *FMIXSUR* as original mixsur/usurp input file
        
    -o OMIXSUR
        Create output file called *OMIXSUR*
        
    -s FSPLITMX
        Use *FSPLITMX* to determine which grids have walls

:Versions:
    * 2017-04-28 ``@ddalle``: First version
"""

# Import relevant modules
import os, sys
import numpy as np
# Command-line parser
import cape.argread
# Utilities: split by comma or space(s)
from cape.util import SplitLineGeneral

# Read a line of a MIXSUR file...
def readline(f):
    """Read a non-blank line from a CGT-like input file
    
    :Call:
        >>> line, V = readline(f)
    :Inputs:
        *f*: :class:`file`
            Open file handle
    :Outputs:
        *line*: :class:`str`
            Raw line from file
        *V*: :class:`list` (:class:`str`) | ``None``
            List of substrings split by commas or spaces, or EOF
    :Versions:
        * 2017-04-28 ``@ddalle``: First version
    """
    # Initialize output
    V = []
    # Read the line
    line = f.readline()
    # Exit if empty
    if line == "": return line, None
    # Split it
    return line, SplitLineGeneral(line)

# Main function
def MixsurGetWalls(*a, **kw):
    """Renumber grids in a MIXSUR input file to skip those without walls
    
    :Call:
        >>> MixsurGetWalls(fmixsur, fsplitmx, omixsur)
        >>> MixsurGetWalls(i="mixsur.i", s="splitmx.i", o="mixsur.surf.i")
    :Inputs:
        *fmixsur*, *i*: {``"mixsur.i"``} | :class:`str`
            Name of original MIXSUR/USURP input file
        *fsplitmx*, *s*: {``"splitmx.i"``} | :class:`str`
            Name of SPLITMX/SPLITMQ input file used to select valid grids
        *omixsur*, *o*: {``"mixsur.surf.i"``} | :class:`str`
            name of modified MIXSUR/USURP input file
    :Versions:
        * 2017-04-28 ``@ddalle``: First version
    """
   # --- Inputs ---
    # Process default name of mixsur input file based on argument
    if len(a) < 1:
        fmixsur = "mixsur.i"
    else:
        fmixsur = a[0]
    # Process default name of splitmx input file base on args
    if len(a) < 2:
        fsplitmx = "splitmx.i"
    else:
        fsplitmx = a[1]
    # Process default name of mixsur modified file
    if len(a) < 3:
        omixsur = "mixsur.surf.i"
    else:
        omixsur = a[2]
    # Use keywords
    fmixsur  = kw.get('i', fmixsur)
    fsplitmx = kw.get('s', fsplitmx)
    omixsur  = kw.get('o', omixsur)
    
   # --- SPLITMX ---
    # Read splitmix file
    G = np.loadtxt(fsplitmx, usecols=(0,), skiprows=2, delimiter=",")
    # Convert to list
    G = list(G)
    
   # --- MIXSUR headers ---
    # Open the input and output files
    fi = open(fmixsur)
    fo = open(omixsur, 'w')
    
    # Read the first line and copy it
    line, V = readline(fi)
    fo.write(line)
    # Read the number of references from the second line
    line, V = readline(fi)
    fo.write(line)
    NREF = int(V[0])
    # Copy the line and the next *NREF* lines
    for i in range(NREF):
        fo.write(fi.readline())
    # Read number of surfs
    line, V = readline(fi)
    fo.write(line)
    # Get NSURF
    NSURF = int(V[0])
    
    # Blank line
    fo.write(fi.readline())
    
   # --- Process Surfs ---
    # Loop through surfaces
    for n in range(NSURF):
        # Read the number of grids
        line, V = readline(fi)
        NSUB = int(V[0])
        # Write the line
        fo.write(line)
        # Loop through the references
        for k in range(NSUB):
            # Read the line
            line = fi.readline()
            # Get the grid number
            V = line.split()
            ng = int(V[0].strip().strip(','))
            # Check for the grid
            V[0] = str(G.index(ng)+1)
            # Write the corrected line
            fo.write(' '.join(V) + "\n")
        # Read prisms
        line, V = readline(fi)
        NPRI = int(V[0])
        # Write the line
        fo.write(line)
        # Loop through the prisms
        for k in range(NPRI):
            fo.write(fi.readline())
        # Blank line
        fo.write(fi.readline())
    
   # --- Cleanup ---
    # Copy the rest of the file
    fo.write(fi.read())
    # Close files
    fi.close()
    fo.close()


# Check if run as a script
if __name__ == "__main__":
    # Process the command-line itnerface inputs
    a, kw = cape.argread.readkeys(sys.argv)
    # Check for a help option
    if kw.get('h') or kw.get('help'):
        import cape.text
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Run the main function
    MixsurGetWalls(*a, **kw)

