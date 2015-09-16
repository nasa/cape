"""
Sectional Loads Module: :mod:`pyCart.lineLoad`
==============================================

This module contains functions for reading and processing sectional loads.  It
is a submodule of :mod:`pyCart.dataBook`.

:Versions:
    * 2015-09-15 ``@ddalle``: Started
"""

# File interface
import os
# Basic numerics
import numpy as np

# Finer control of dicts
from .options import odict
# Utilities or advanced statistics
from . import util


# Line loads
class DBLineLoad(object):
    """Data book line load class
    
    """
    


# Line loads for a case
class CaseLL(object):
    """Line load class
    
    
    :Versions:
        * 2015-09-25 ``@ddalle``: First version
    """
    
    def __init__(self, comp="entire"):
        """Initialization method"""
        # Read the derivative loads
        d = self.ReadLDS('LineLoad_%s.dlds' % comp)
        # Save the cut coordinates
        self.x = d['x']
        # Save the force contributions
        self.CA = d['CA']
        self.CY = d['CY']
        self.CN = d['CN']
        # Save the moment derivatives
        self.CLL = d['CLL']
        self.CLM = d['CLM']
        self.CLN = d['CLN']
    
    # Function to read a file
    def ReadLDS(fname):
        """Read a sectional loads ``*.?lds`` from `triloadCmd`
        
        :Call:
            >>> d = LL.ReadLDS(fname)
        :Inputs:
            *LL*: :class:`pyCart.lineLoad.CaseLL`
                Instance of case line load
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *d*: :class:`dict` (:class:`numpy.ndarray`)
                Dictionary of each line load coefficient
            *d['x']*: :class:`numpy.ndarray`
                Vector of cut coordinates
            *d['CA']*: :class:`numpy.ndarray`
                Axial force contribution
        :Versions:
            * 2015-09-15 ``@ddalle``: First version
        """
        # Open the file.
        f = open(fname, 'r')
        # Read lines until it is not a comment.
        line = '#'
        while (not line.lstrip().startswith('#')) and (len(line)>0):
            # Read the next line.
            line = f.readline()
        # Exit if empty.
        if len(line) == 0:
            return {}
        # Number of columns
        nCol = len(line.split())
        # Go backwards one line from current position.
        f.seek(-len(line), 1)
        # Read the rest of the file.
        D = np.fromfile(f, count=-1, sep=' ')
        # Reshape to a matrix
        D = D.reshape((D.size/nCol, nCol))
        # Save the keys.
        return {
            'x':   D[:,0],
            'CA':  D[:,1],
            'CY':  D[:,2],
            'CN':  D[:,3],
            'CLL': D[:,4],
            'CLM': D[:,5],
            'CLN': D[:,6]
        }
        
            
