#!/usr/bin/env python
"""
General force, moment, or residual history plot: :mod:`pc_Plot`
===============================================================

Plot *CA*, *CY*, *CN*, and the L1 residual for 

:Call:

    .. code-block:: bash
    
        $ pc_Plot4.py $COMP [OPTIONS]
        $ pc_Plot4.py $COMP1 $COMP2 [...] [OPTIONS]

:Inputs:
    * *COMP*: Name of component to plot forces and moments on {entire}
    
:Options:
    -h, --help
        Display this help message and exit
        
    -i
        Interactive; displays the plot after saving it
        
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
    a, kw = pyCart.argread.readkeys(sys.argv)
    # Set components.
    C = ['CA', 'CY', 'CN', 'L1']
    # Set the plot layout.
    kw['nRow'] = 2
    kw['nCol'] = 2
    # Check for help flag.
    if kw.get('h') or kw.get('help'):
        # Display help message and exit.
        print(__doc__)
        sys.exit()
    # Default component.
    if a == []:
        a = ['entire']
    # Loop through input components.
    for comp in a:
        # Set the component.
        kw['p'] = comp
        # Do the plotting.
        pyCart.pc_Plot.pc_Plot(C, **kw)
    
