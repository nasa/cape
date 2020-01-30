#!/usr/bin/env python

# System modules
import os
import copy
import numpy as np


# Initialization settings
def InitNAC1(cntl):
    """Modify complex settings that should always be applied

    For instance, sets *ArchiveFolder* according to user

    :Call:
        >>> InitNAC(cntl)
    :Inputs:
        *cntl*: :class:`pyOver.overflow.Overflow`
            OVERFLOW settings interface
    """

# Filter options based on the *Label* trajectory key
def ApplyInletBC(cntl, v, i):
    """Modify BCINP for nacelle inlet face

    This method is modifies the BCINP namelist in the OVERFLOW input file 
    for the boundary conditions on the Inlet grid

    :Call:
        >>> ApplyInletBC(cntl, v, i)
    :Inputs:
        *cntl*: :class:`pyOver.overflow.Overflow`
            OVERFLOW settings interface
        *i*: :class:`int`
            Case number
    :Versions:
        * 2020-01-30 ``@serogers``: First version
    """

    pass
    # Get the specified label
    #lbl = cntl.x['Label'][i]


    ## Inlet grid: set boundary conditions
    #grid = 'Inlet'
    #bci = 3
    ## Test case 1:
    #print("\nLabel = %s" % lbl)
    #if 'test01' in lbl:
    #    ibtyp = [5, 16, 33, 22]
    #    bcpar1 = 6.666
    #    bcpar2 = 7.234
    #    bcfile = None
    #elif 'test02' in lbl:
    #    ibtyp = [5, 16, 34, 22]
    #    bcpar1 = 6.012
    #    bcpar2 = 7.123
    #    bcfile = 'INLET BC'
    #elif 'test03' in lbl:
    #    ibtyp = [5, 16, 34, 22]
    #    bcpar1 = 0
    #    bcpar2 = None
    #    bcfile = 'INLET BC'
    ## Now apply the boundary conditions for Inlet
    #cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'IBTYP', ibtyp)
    #cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCPAR1', bcpar1, i=bci)
    #cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCPAR2', bcpar2, i=bci)
    #cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCFILE', bcfile, i=bci)

    ## Exit grid: set boundary conditions
    #grid = 'Exit'
    #bci = 3
    ## Test case 1, 2, and 3:
    #if lbl == 'test01' or lbl == 'test02' or lbl == 'test03':
    #    ibtyp = [5, 16, 141, 22]
    #    bcpar1 = 1.2
    #    bcpar2 = 1.385
    #    bcfile = None
    ## Now apply the boundary conditions for Inlet
    #cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'IBTYP', ibtyp)
    #cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCPAR1', bcpar1, i=bci)
    #cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCPAR2', bcpar2, i=bci)
    #cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCFILE', bcfile, i=bci)
    #    

    ######################333
    ## DEBUG: report IBTYP for Inlet
    #ibtyp  = cntl.Namelist.GetKeyFromGrid('Inlet', 'BCINP', 'IBTYP')
    #bcpar1 = cntl.Namelist.GetKeyFromGrid('Inlet', 'BCINP', 'BCPAR1', i=bci)
    #bcpar2 = cntl.Namelist.GetKeyFromGrid('Inlet', 'BCINP', 'BCPAR1', i=bci)
    #print("In ApplyLabel:  IBTYP,BCPAR1,BCPAR2 for Inlet = ", 
    #      ibtyp, bcpar1, bcpar2)
    ######################333


# def ApplyBCINP


