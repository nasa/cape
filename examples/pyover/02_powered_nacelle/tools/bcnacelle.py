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

    ## Inlet grid: set boundary conditions
    grid = 'Inlet'
    bci = 3
    print("\n\nIn function ApplyInletBC, v = ", v)
    # Extract the BCINP from the template for this grid
    IBTYP = cntl.Namelist.GetKeyFromGrid(grid, 'BCINP', 'IBTYP')

    #################################################
    # Process the pressure BC
    if IBTYP.count(33) > 0:
        # Get the column for ibtyp=33
        bci = IBTYP.index(33)
        # Change bci to 1-based index
        bci += 1
        # Set the BCPAR1 value for this case
        cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCPAR1', v, i=bci)
        BCPAR1 = cntl.Namelist.GetKeyFromGrid(grid, 'BCINP', 'BCPAR1', i=bci)

# def ApplyInletBC

def ApplyExitBC(cntl, v, i):
    """Modify BCINP for nacelle exit face

    This method is modifies the BCINP namelist in the OVERFLOW input file 
    for the boundary conditions on the Exit grid

    The IBTYP=141 boundary condition applies a plug-nozzle inflow BC,
    where the total pressure and temperature ratios are specified
    in BCPAR1 and BCPAR2, respectively.


    :Call:
        >>> ApplyExitBC(cntl, v, i)
    :Inputs:
        *cntl*: :class:`pyOver.overflow.Overflow`
            OVERFLOW settings interface
        *i*: :class:`int`
            Case number
    :Versions:
        * 2020-02-05 ``@serogers``: First version
    """

    ## Exit grid: set boundary conditions
    grid = 'Exit'
    print("In function ApplyExitBC, v = ", v)
    # Extract the BCINP from the template for this grid
    IBTYP = cntl.Namelist.GetKeyFromGrid(grid, 'BCINP', 'IBTYP')

    #################################################
    # Process the nozzle info BC
    if IBTYP.count(141) > 0:
        # Get the column for ibtyp=141
        bci = IBTYP.index(141)
        # Change bci to 1-based index
        bci += 1
        # Set the BCPAR1 value for this case: controls total pressure
        cntl.Namelist.SetKeyForGrid(grid, 'BCINP', 'BCPAR1', v, i=bci)
        BCPAR1 = cntl.Namelist.GetKeyFromGrid(grid, 'BCINP', 'BCPAR1', i=bci)

