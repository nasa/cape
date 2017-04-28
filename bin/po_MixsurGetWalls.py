#!/usr/bin/env python
"""
Renumber MIXSUR Input Subsets to Account for No-Wall Grids
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
    
    # Read splitmix file
    G = np.loadtxt(fsplitmx, usecols=(0,), skiprows=2, delimiter=",")
    
    # Open the input and output files
    fi = open(fmixsur)
    fo = open(omixsur, 'w')
    
    
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

