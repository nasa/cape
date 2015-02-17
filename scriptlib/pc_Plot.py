#!/usr/bin/env python
"""
General force, moment, or residual history plot: :mod:`pc_Plot`
===============================================================

Plot a specified list of coefficients.  The coefficients can be any from the
following list.

    =====  ============================
    Coeff  Description
    =====  ============================
    *CA*   Axial force coefficient
    *CY*   Lateral force coefficient
    *CN*   Normal force coefficient
    *CLL*  Rolling moment coefficient
    *CLM*  Pitching moment coefficient
    *CLN*  Yawing moment coefficient
    *L1*   Total L1 residual (density)
    =====  ============================

:Call:

    .. code-block:: console
    
        $ pc_Plot.py $C1 $C2 [...] -p $COMP [OPTIONS]

:Inputs:
    * *C1*: Name of first coefficient to plot
    * *C2*: Name of second coefficient
    * *COMP*: Name of component to plot
    
:Options:
    -h, --help
        Display this help message and exit
        
    -i
        Interactive; displays the plot after saving it
        
    -p COMP
        Plot the forces and moments on *COMP* [{entire} | :class:`str`]
        
    -f FNAME
        Use settings from file *FNAME* [{plot.json} | :class:`str`]
        
    --nRow NROW
        Use *NROW* rows in the plot
        
    --nCol NCOL
        Use *NCOL* columns of subfigures in the plot
        
    --ext EXT
        Save figure as *EXT* format [{pdf} | svg | png]
        
    --dpi DPI
        Use *DPI* dots per inch if saving to a raster format
        
    -o FNAME
        Save plot to file *FNAME*; overrides default and extension
        
    -n NSHOW, --nShow NSHOW
        Show only the last *NSHOW* iterations in plots
    
    --nAvg NAVG
        Calculate averages using last *NAVG* iterations for forces and moments
        
    -d D, -d
        Use *D* to show spread from mean with dotted lines, or turn off with -d
        
    --dCA DCA
        Use *DCA* as spread on C_A plots; also for other coefficients
        
    --SBU
        Display 'SBU - ITAR' at the bottom of plot
        
    --restriction FSBU
        Use string *FSBU* as sensitivity tag and display at bottom of plot
        
    --tag FTAG
        Print *FTAG* in upper left corner of plot
        
:Versions:
    * 2014-11-13 ``@ddalle``: First version
"""

# Modules
import pyCart.argread
import pyCart.pc_Plot
# System interface.
import sys
        
    
# Check for running as a script.
if __name__ == "__main__":
    # Process inputs.
    C, kw = pyCart.argread.readkeys(sys.argv)
    # Check for help flag.
    if kw.get('h') or kw.get('help'):
        # Display help message and exit.
        print(__doc__)
        sys.exit()
    # Do the plotting.
    pyCart.pc_Plot.pc_Plot(C, **kw)
    
