"""
Data Book Module: :mod:`pyCart.dataBook`
========================================

This module contains functions for reading and processing forces, moments, and
other statistics from a trajectory

:Versions:
    * 2014-12-20 ``@ddalle``: Started
"""

# File interface
import os
# Basic numerics
import numpy as np
# Advanced text (regular expressions)
import re

#<!--
# ---------------------------------
# I consider this portion temporary

# Get the umask value.
umask = 0027
# Get the folder permissions.
fmask = 0777 - umask
dmask = 0777 - umask

# ---------------------------------
#-->


# Aerodynamic history class
class DataBook(dict):
    """
    This class provides an interface to the data book for a given CFD run
    matrix.
    
    :Call:
        >>> DB = pyCart.dataBook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`pyCart.trajectory.Trajectory`
            The current pyCart trajectory (i.e. run matrix)
        *opts*: :class:`pyCart.options.Options`
            Options class for interrogating data book options
    :Outputs:
        *DB*: :class:`pyCart.dataBook.DataBook`
            Instance of the pyCart data book class
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    # Initialization method
    def __init__(self, x, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Save the root directory.
        self.RootDir = os.getcwd()
        # Save the components
        self.Components = opts.get_DataBookComponents()
        # Save the folder
        self.Dir = opts.get_DataBookDir()
        # Make sure the destination folder exists.
        for fdir in self.Dir.split('/'):
            # Check if the folder exists.
            if not os.path.isdir(fdir):
                os.mkdir(fdir, dmask)
            # Go to the folder.
            os.chdir(fdir)
        # Go back to root folder.
        os.chdir(self.RootDir)
        # Loop through the components.
        for comp in self.Components:
            # Initialize the data book.
            self[comp] = DBComp(comp, x, opts, fdir)
        
                
                
# Individual component data book
class DBComp(dict):
    """
    Individual component data book
    
    :Call:
        >>> DBi = DBComp(comp, x, opts, fdir="data")
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *x*: :class:`pyCart.trajectory.Trajectory`
            Trajectory for processing variable types
        *opts*: :class:`pyCart.options.DataBook`
            Options for the component
        *fdir*: :class:`str`
            Data book folder (forward slash separators)
    :Outputs:
        *DBi*: :class:`pyCart.dataBook.DBComp`
            An individual component data book
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    # Initialization method
    def __init__(self, comp, x, opts, fdir="data"):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # DataBook delimiter
        delim = opts.get_Delimiter()
        # Get the list of columns for that coefficient.
        cols = opts.get_DataBookCols(comp)
        
        # Construct the file name.
        fcomp = 'aero_%s.dat' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        # Check for the file.
        if os.path.isfile(fname):
            # Initialize column number
            nCol = 0
            # Loop through trajectory keys.
            for k in x.keys:
                # Get the type.
                t = x.defns[k].get('Value', 'float')
                # Read the column
                self[k] = np.loadtxt(fname, 
                    delimiter=delim, dtype=t, usecols=nCol)
                # Increase the column number
                nCol += 1
            # Loop through the data book columns.
            for c in cols:
                # Add the column.
                self[c] = np.loadtxt(fname, delimiter=delim, usecols=nCol)
                # Increase column number.
                nCol += 1
        else:
            # Initialize empty trajectory arrays.
            for k in x.keys:
                self[k] = np.array([], dtype=x.defns[k].get('Value', 'float'))
            # Initialize the data columns.
            for c in cols:
                self[c] = np.array([])
        
        
        
