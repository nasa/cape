"""
This module imports the generic utilities using

    .. code-block:: python

        from cape.util import *

It also stores the absolute path to the folder containing the
:mod:`cape.pycart` module as the variable *pyCartFolder*.

The module also provides several other methods for reading multiple
files to determine the situational status of the Cart3D solution in the
present working directory. These methods may be duplicated in
:mod:`cape.pycart.case`.

:See also:
    * :mod:`cape.util`
"""

# Standard library
import os
import glob
import subprocess as sp

# Local imports
from ..util import *


# pyCart base folder
pyCartFolder = os.path.split(os.path.abspath(__file__))[0]


# Function to get the most recent working folder
def GetWorkingFolder():
    r"""Get the most recent working folder

    Can be one of the following:

        * ``.`` (present directory)
        * ``adapt??``
        * ``adapt??/FLOW``

    This function must be called from the top level of a case run
    directory.

    :Call:
        >>> fdir = GetWorkingFolder()
    :Outputs:
        *fdir*: :class:`str`
            Name of the most recently used working folder with a history file
    :Versions:
        * 2014-11-24 ``@ddalle``: v1.0
        * 2023-06-06 ``@ddalle``: v2.0; support ``adapt??/FLOW/``
    """
    # Initialize working directory.
    fdir = '.'
    # Implementation of returning to adapt after startup turned off
    if os.path.isfile('history.dat') and not os.path.islink('history.dat'):
        return fdir
    # Check for adapt?? folders
    fglob = glob.glob('adapt??')
    fglob.sort()
    fglob.reverse()
    # Check adapt?? folders in reverse
    for fi in fglob:
        # Candidate folders
        d1 = fi
        d2 = os.path.join(fi, "FLOW")
        # Candidate files
        f1 = os.path.join(d1, "history.dat")
        f2 = os.path.join(d2, "history.dat")
        # Search *d2* first
        fdirs = (d2, d1)
        fnams = (f2, f1)
        # Search
        for dj, fj in zip(fdirs, fnams):
            # Check if file exists
            if os.path.isfile(fj):
                return dj
    # No adapt??/{FLOW/}history.dat file: use base folder
    return fdir


# Function to read last line of 'history.dat' file
def GetHistIter(fname='history.dat'):
    r"""Get the most recent iteration number from a ``history.dat`` file

    :Call:
        >>> n = GetHistIter(fname='history.dat')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *n*: :class:`float`
            Last iteration number
    :Versions:
        * 2014-11-24 ``@ddalle``: v1.0 (``cape.cfdx.util``)
        * 2015-12-04 ``@ddalle``: v1.0
    """
    # Check the file beforehand.
    if not os.path.isfile(fname):
        # No history
        return 0
    # Check the file.
    try:
        # Try to tail the last line.
        txt = sp.Popen(['tail', '-1', fname], stdout=sp.PIPE).communicate()[0]
        # Try to get the integer.
        return float(txt.split()[0])
    except Exception:
        # Read the last line.
        line = open(fname).readlines()[-1]
        # Get iteration string
        txt = line.split()[0]
        # Return iteration number
        return float(txt)


# Get steady-state history iteration
def GetSteadyHistIter():
    r"""Get largest steady-state iteration number from ``history.dat``

    :Call:
        >>> n = GetSteadyHistIter()
    :Outputs:
        *n*: :class:`int`
            Iteration number of last line w/o decimal (integer)
    :Versions:
        * 2015-12-02 ``@ddalle``: v1.0
    """
    # Candidate history files
    f1 = 'history.dat'
    f2 = os.path.join('BEST', 'history.dat')
    f3 = os.path.join('BEST', 'FLOW', 'history.dat')

    # Get the history file.
    if os.path.isfile(f1):
        # Standard working folder
        fname = f1
    elif os.path.isfile(f2):
        # Adaptive working folder
        fname = f2
    elif os.path.isfile(f3):
        # Version 1.5+ adaptive working folder
        fname = f3
    else:
        # None.
        return 0
    # Call GREP if possible
    try:
        # Get the last steady-state iteration number.
        txt = sp.Popen(
            [r'egrep "^\s+[0-9]+ " %s | tail -n 1' % fname],
            shell=True, stdout=sp.PIPE).communicate()[0]
        # Get the first entry, which is the iteration number.
        return int(txt.split()[0])
    except Exception:
        # Initialize.
        n = 0
        line = '\n'
        # Open the file.
        f = open(fname, 'r')
        # Loop through lines until unsteady iteration or eof
        while line != '':
            # Read next line.
            line = f.readline()
            # Check comment.
            if line.startswith('#'):
                continue
            # Check for decimal.
            try:
                if '.' in line.split()[0]:
                    break
            except Exception:
                break
            # Read iteration number
            n = int(line.split()[0])
        # Output
        return n


# Get unsteady history iteration
def GetUnsteadyHistIter():
    r"""Get largest time-accurate iteration number from ``history.dat``

    :Call:
        >>> n = GetUnsteadyHistIter()
    :Outputs:
        *n*: :class:`float`
            Most recent iteration number, including partial iterations
    :Versions:
        * 2015-12-02 ``@ddalle``: v1.0
    """
    # Candidate history files
    f1 = 'history.dat'
    f2 = os.path.join('BEST', 'history.dat')
    f3 = os.path.join('BEST', 'FLOW', 'history.dat')

    # Get the history file.
    if os.path.isfile(f1):
        # Standard working folder
        fname = f1
    elif os.path.isfile(f2):
        # Adaptive working folder
        fname = f2
    elif os.path.isfile(f3):
        # Version 1.5+ adaptive working folder
        fname = f3
    else:
        # None.
        return 0
    # Call GREP if possible
    try:
        # Get the last steady-state iteration number.
        txt = sp.Popen(
            ['tail', '-1', fname],
            stdout=sp.PIPE).communicate()[0]
        # Get the first entry, which is the iteration number.
        txt0 = txt.split()[0]
        # Check for unsteady.
        if '.' in txt0:
            # Ends with a time-accurate iteration
            return float(txt0)
        else:
            # Ends with a steady-state iteration
            return 0
    except Exception:
        # Read the last line.
        line = open(fname).readlines()[-1]
        # Get iteration string
        txt = line.split()[0]
        # Check for decimal
        if '.' not in txt: return 0
        # Return iteration number
        return float(txt)


# Get total history iteration
def GetTotalHistIter():
    r"""Get current iteration from ``history.dat`` corrected by restart

    :Call:
        >>> n = GetUnsteadyHistIter()
    :Outputs:
        *n*: :class:`float`
            Most recent iteration number, including partial iters
    :Versions:
        * 2015-12-02 ``@ddalle``: v1.0
    """
    # Return history
    return GetSteadyHistIter() + GetUnsteadyHistIter()

