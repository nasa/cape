#!/usr/bin/env python
"""
Convert FUN3D Tecplot file to Cart3D triangulation: ``pf_Plt2Triq.py``
======================================================================

Convert a Tecplot ``.plt`` file from FUN3D 

:Call:

    .. code-block:: console
    
        $ pf_Plt2Triq.py $PLT [$TRIQ OPTIONS]
        $ pf_Plt2Triq.py -h

:Inputs:
    * *PLT*: Name of FUN3D Tecplot '.plt' file
    * *TRIQ*: Name of output '.triq' file
    
:Options:
    -h, --help
        Display this help message and exit
        
    -i PLT
        Use *PLT* as input file
        
    -o TRIQ
        Use *TRIQ* as name of created output files
        
    -q
        Use all state variables in order instead of extracting best state
        variables for ``triloadCmd``
        
    --avg 0
        Do not use time average, even if available
        
    --rms
        Write RMS of each variable instead of nominal/average value
    
If the name of the output file is not specified, it will just add ``triq`` as
the extension to the input (deleting ``.plt`` if possible).

:Versions:
    * 2016-12-19 ``@ddalle``: First version
"""

# Module to handle inputs and os interface
import sys
import numpy as np
# Command-line input parser
import cape.argread as argr
# Get the Tecplot module.
import pyFun.plt

# Main function
def Plt2Triq(*a, **kw):
    """
    Convert a UH3D triangulation file to Cart3D tri format
    
    :Call:
        >>> Plt2Triq(plt, triq, **kw)
        >>> UH3D2Tri(i=fplt, o=ftriq, **kw)
    :Inputs:
        *fplt*: :class:`str`
            Name of input file
        *ftriq*: :class:`str`
            Name of output file (default: replace extension with ``.triq``)
        *ascii*: {``True``} | ``False``
            Write ASCII triq file?
    :Versions:
        * 2016-12-19 ``@ddalle``: First version
    """
    # Get the file name.
    if len(a) == 0:
        # Defaults
        fplt = None
    else:
        # Use the first general input.
        fplt = a[0]
    # Prioritize a "-i" input.
    fplt = kw.get('i', fplt)
    # Must have a file name.
    if fplt is None:
        # Required input.
        print __doc__
        raise IOError("At least one input required.")
    # Options
    triload = not kw.get('q', False)
    tavg    = kw.get('avg', True)
    trms    = kw.get('rms', False)
        
    # Read in the plt file.
    plt = pyFun.plt.Plt(fplt)
    # Create the triq interface
    triq = plt.CreateTriq(triload=triload, avg=tavg, rms=trms)
    
    # Get file extension
    ext = triq.GetOutputFileType(**kw)
    # Default file name
    if ext == 'ascii':
        # ASCII file: use ".tri"
        ftriq = fplt.rstrip('plt').rstrip('dat') + 'triq'
    else:
        # Binary file: use ".i.tri"
        ftriq = fplt.rstrip('plt').rstrip('dat') + 'i.triq'
    # Get the output file name if given as second input
    if len(a) >= 2: ftriq = a[1]
    # Prioritize a "-i" input.
    ftriq = kw.get('o', ftriq)
    
    # Write triangulation
    triq.Write(ftriq, **kw)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function.
    Plt2Triq(*a, **kw)
    
