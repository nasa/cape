"""
Utilities for pyCart: :mod:`pyCart.util`
========================================

"""

# Import everything from cape.util
from cape.util import *


# pyCart base folder
pyCartFolder = os.path.split(os.path.abspath(__file__))[0]

# Get steady-state history iteration
def GetSteadyHistIter():
    """Get largest steady-state iteration number from ``history.dat``
    
    :Call:
        >>> n = GetSteadyHistIter()
    :Outputs:
        *n*: :class:`int`
            Iteration number of last line without decimal in iteration number
    :Versions:
        * 2015-12-02 ``@ddalle``: First version
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
        txt = sp.Popen(['egrep "^\s+[0-9]+ " %s | tail -n 1' % fname],
            shell=True).communicate()[0]
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
            if line.startswith('#'): continue
            # Check for decimal.
            if '.' in line.split()[0]: break
            # Read iteration number
            n = int(line.split()[0])
        # Output
        return n
        
# Get unsteady history iteration
def GetUnsteadyHistIter():
    """Get largest time-accurate iteration number from ``history.dat``
    
    :Call:
        >>> n = GetUnsteadyHistIter()
    :Outputs:
        *n*: :class:`float`
            Iteration number of last line with decimal in iteration number
    :Versions:
        * 2015-12-02 ``@ddalle``: First version
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
        txt = sp.Popen(['tail -n 1 %s' % fname],
            shell=True).communicate()[0]
        # Get the first entry, which is the iteration number.
        return float(txt.split()[0])
    except Exception:
        # Read the last line.
        line = f.readlines()[-1]
        # Get iteration string
        txt = line.split()[0]
        # Check for decimal
        if '.' not in txt: return 0
        # Return iteration number
        return float(txt)
        
# Get total history iteration
def GetTotalHistIter():
    """Get current iteration from ``history.dat`` corrected by restart
    
    :Call:
        >>> n = GetUnsteadyHistIter()
    :Outputs:
        *n*: :class:`float`
            Iteration number of last line with decimal in iteration number
    :Versions:
        * 2015-12-02 ``@ddalle``: First version
    """
    # Return history
    return GetSteadyHistIter() + GetUnsteadyHistIter()

