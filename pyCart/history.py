"""
Aerodynamic Iterative History: :mod:`pyCart.aero`
=================================================

This module contains functions for reading and processing forces, moments, and
other statistics from a run directory.

:Versions:
    * 2014-11-12 ``@ddalle``: Starter version
"""

# File interface
import os
# Basic numerics
import numpy as np


    

# Aerodynamic history class
class History(object):
    """
    Iterative history class
    
    This class provides an interface to residuals, CPU time, and similar data
    for a given run directory
    
    :Call:
        >>> hist = pyCart.history.History()
    :Outputs:
        *hist*: :class:`pyCart.history.History`
            Instance of the run history class
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
    
    # Initialization method
    def __init__(self):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        
        # Process the best data folder.
        if os.path.isfile('history.dat'):
            # Subsequent non-adaptive runs
            fdir = '.'
        elif os.path.islink('BEST'):
            # There's a BEST/ folder; use it as most recent adaptation cycle.
            fdir = 'BEST'
        elif os.path.isdir('adapt00'):
            # It's an adaptive run, but it hasn't gotten far yet.
            fdir = 'adapt00'
        else:
            # This is not an adaptive cycle; use root folder.
            fdir = '.'
        # History file name.
        fhist = os.path.join(fdir, 'history.dat')
        # Read the file.
        lines = open(fhist).readlines()
        # Filter comments.
        lines = [l for l in lines if not l.startswith('#')]
        # Convert all the values to floats.
        A = np.array([[float(v) for v in l.split()] for l in lines])
        # Get the indices of steady-state iterations.
        # (Time-accurate iterations are marked with decimal step numbers.)
        i = np.array(['.' not in l.split()[0] for l in lines])
        # Check for steady-state iterations.
        if np.any(i):
            # Get the last steady-state iteration.
            n0 = np.max(A[i,0])
            # Add this to the time-accurate iteration numbers.
            A[np.logical_not(i),0] += n0
        # Eliminate subiterations.
        A = A[np.mod(A[:,0], 1.0) == 0.0]
        # Save the number of iterations.
        self.nIter = A.shape[0]
        # Save the iteration numbers.
        self.i = np.arange(self.nIter) + 1
        # Save the CPU time per processor.
        self.CPUtime = A[:,1]
        # Save the maximum residual.
        self.maxResid = A[:,2]
        # Save the global residual.
        self.L1Resid = A[:,3]
        # Check for a 'user_time.dat' file.
        if os.path.isfile('user_time.dat'):
            # Initialize time
            t = 0.0
            # Loop through lines.
            for line in open('user_time.dat').readlines():
                # Check comment.
                if line.startswith('#'): continue
                # Add to the time.
                t += np.sum([float(v) for v in line.split()[1:]])
        else:
            # Find the indices of run break points.
            i = np.where(self.CPUtime[1:] < self.CPUtime[:-1])[0]
            # Sum the end times.
            t = np.sum(self.CPUtime[i]) + self.CPUtime[-1]
        # Save the time.
        self.CPUhours = t / 3600.


