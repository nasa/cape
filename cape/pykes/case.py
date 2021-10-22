#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.case`: Kestrel individual-case module
=======================================================

This module provides interfaces to run a single case of Kestrel from
the case folder. It also provides tools such as :func:`GetCurrentIter`,
which determines how many cases have been run.

The functions of this module expect to be run from a Kestrel case folder
created by :mod:`cape.pykes`.

"""

# Standard library
import glob
import os
import sys

# Third-party


# Local imports
from . import cmdgen
from .. import argread
from .. import text as textutils
from ..cfdx import case as cc
from ..cfdx import queue
from .optoins.runcontrol import RunControl



# Help message for CLI
HELP_RUN_KESTREL = r"""
``run_kestrel.py``: Run Kestrel for one phase
================================================

This script determines the appropriate phase to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:
    
    .. code-block:: console
    
        $ run_kestrel.py [OPTIONS]
        $ python -m cape.pykes run [OPTIONS]
        
:Options:
    
    -h, --help
        Display this help message and quit

:Versions:
    * 2021-10-21 ``@ddalle``: Version 1.0
"""


# Function to run Kestrel for one phase (and restart if approp.)
def run_kestrel():
    r"""Setup and run the appropriate Kestrel command

    This function runs one phase, but may restart the case by recursing
    if settings prescribe it.

    :Call:
        >>> run_kestrel()
    :Versions:
        * 2021-10-21 ``@ddalle``: Version 0.1; started
    """
    # Process arguments
    a, kw = argread.readkeys(sys.argv)
    # Check for help argument.
    if kw.get('h') or kw.get('help'):
        # Display help and exit
        print(textutils.markdown(HELP_RUN_KESTREL))
        return
    # Start RUNNING and initialize timer
    tic = cc.init_timer()
    # Read settings
    rc = read_case_json()
    # Get phase number
    j = get_phase(rc)
    # Write the start time
    write_starttime(tic, rc, j)
    # Prepare files
    prepare_files(rc, j)


# --- STATUS functions ---
def get_phase(rc):
    r"""Determine the phase number based on files in folder
    
    :Call:
        >>> j = get_phase(rc)
    :Inputs:
        *rc*: :class:`RunControl`
            Case *RunControl* options
    :Outputs:
        *j*: :class:`int`
            Most appropriate phase number for a (re)start
    :Versions:
        * 2021-10-21 ``@ddalle``: Version 1.0
    """
    # Get the iteration from which a restart would commence
    n = None
    # Start with phase 0 if ``None``
    if n is None:
        return rc.get_PhaseSequence(0)
    # Get last phase number
    j = rc.get_PhaseSequence(-1)
    # Special check for --skeleton cases
    if len(glob.glob("run.%02i.*" % j)) > 0:
        # Check iteration count
        if n >= rc.get_PhaseIters(j):
            return j
    # Loop through phases
    for j in rc.get_PhaseSequence():
        # Target iterations for this phase
        nt = rc.get_PhaseIters(j)
        # Check output files
        if len(glob.glob("run.%02i.*" % j)) == 0:
            # This phase has not been run
            return j
        # Check the iteration- numbers
        if nt is None:
            # Don't check null phases
            pass
        elif n < nt:
            # Case has been run but hasn't reached target
            return j
    # Case completed; just return the last phase
    return j



# Function to read the local settings file
def read_case_json():
    r"""Read *RunControl* settings from ``case.json``
    
    :Call:
        >>> rc = read_case_json()
    :Outputs:
        *rc*: :class:`cape.pykes.options.runcontrol.RunControl`
            Case run control settings
    :Versions:
        * 2021-10-21 ``@ddalle``: Version 1.0
    """
    return cc.read_case_json(RunControl)


# Write start time
def write_starttime(tic, rc, j, fname="pykes_start.dat"):
    r"""Write the start time from *tic*
    
    :Call:
        >>> write_starttime(tic, rc, j, fname="pykes_start.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time to write into data file
        *rc*: :class:`RunControl`
            Options interface
        *j*: :class:`int`
            Phase number
        *fname*: {``"pykes_start.dat"``} | :class:`str`
            Name of file containing run start times
    :Versions:
        * 2021-10-21 ``@ddalle``: Version 1.0
    """
    # Call the function from :mod:`cape.cfdx.case`
    cc.WriteStartTimeProg(tic, rc, i, fname, "run_kestrel.py")


