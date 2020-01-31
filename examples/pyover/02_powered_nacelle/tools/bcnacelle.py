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

def ApplyInletBC(cntl, v, i):
    """Modify BCINP for nacelle inlet face

    This method is modifies the BCINP namelist in the OVERFLOW input file 
    for the boundary conditions on the Inlet grid

    The IBTYP=33 boundary condition applies a contant pressure outflow
    at the engine inlet face. This uses the value of BCPAR1 to set the
    ratio of the boundary static pressure to freestream pressure.

    The IBTYP=34 boundary condition applies a constant mass-flow rate
    at the engine inlet face. This uses the value of BCPAR1 to set the
    target mass-flow rate.  BCPAR2 sets the update rate and relaxation factor.
    BCFILE is used to supply the FOMOCO component and Aref.

    :Call:
        >>> ApplyInletBC(cntl, v, i)
    :Inputs:
        *cntl*: :class:`pyOver.overflow.Overflow`
            OVERFLOW settings interface
        *v*: :class:`float`
            Run-matrix value in the InletBC column for case i
        *i*: :class:`int`
            Case number
    :Versions:
        * 2020-01-30 ``@serogers``: First version
    """

    # Get the specified label
    lbl = cntl.x['Label'][i]

    ## Inlet grid: set boundary conditions
    grid = 'Inlet'
    bci = 3
    print("\n\nIn function ApplyInletBC, v = ", v)
    # Extract the BCINP from the template
    IBTYP = cntl.Namelist.GetKeyFromGrid(grid, 'BCINP', 'IBTYP')
    BCPAR1 = cntl.Namelist.GetKeyFromGrid(grid, 'BCINP', 'BCPAR1', i=3)
    print("Existing value of IBTYP = ", IBTYP)
    print("Existing value of BCPAR1(3) = ", BCPAR1)


    #################################################
    # Process the pressure BC
    if IBTYP.count(33) > 0:
        # Get the column for ibtyp=33
        bci = IBTYP.index(33)
        # Change bci to 1-based index
        bci += 1
        # Set the BCPAR1 value for this case
        cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCPAR1', v, i=3)
        BCPAR1 = cntl.Namelist.GetKeyFromGrid(grid, 'BCINP', 'BCPAR1', i=3)
        print("    New value of BCPAR1 = ", BCPAR1)



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
# def ApplyInletBC

def ApplyExitBC(cntl, v, i):
    """Modify BCINP for nacelle exit face

    This method is modifies the BCINP namelist in the OVERFLOW input file 
    for the boundary conditions on the Exit grid

    The IBTYP=141 boundary condition applies TBD


    :Call:
        >>> ApplyExitBC(cntl, v, i)
    :Inputs:
        *cntl*: :class:`pyOver.overflow.Overflow`
            OVERFLOW settings interface
        *i*: :class:`int`
            Case number
    :Versions:
        * 2020-01-31 ``@serogers``: First version
    """

    # Get the specified label
    lbl = cntl.x['Label'][i]

    ## Inlet grid: set boundary conditions
    grid = 'Exit'
    bci = 3
    print("\n\nIn function ApplyExitBC, v = ", v)

