#!/usr/bin/env python
"""
Python interface to Plot3D files
================================

:Versions:
    * 2016-02-26 ``@ddalle``: First version
"""

# Numerics
import numpy as np
# System interface
import os

# General Plot3D class...
class Plot3D(object):
    """General Plot3D class
    
    :Call:
        >>> q = Plot3D(fname, endian=None)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *endian*: ``None`` | ``"big"`` | ``"little"``
            Manually-specified byte-order
    :Outputs:
        *q*: :class:`cape.plot3d.Plot3D`
            Plot3D interface
    :Versions:
        * 2016-02-26 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname, endian=None):
        """Initialization method
        
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Save file name
        self.fname = os.path.abspath(fname)
        # Endianness
        self.endian = endian
        # File handle
        self.f = None
        # Status
        self.closed = True
        # Save reasonable default data types
        self.itype = 'i4'
        self.ftype = 'f8'
    
    # Open the file
    def open(self, mode='rb'):
        """Open the file with the correct mode
        
        :Call:
            >>> q.open(mode='rb')
        :Inputs:
            *q*: :class:`cape.plot3d.Plot3D`
                Plot3D file interface
            *mode*: {'rb'} | 'wb' | 'rb+' | :class:`str`
                File mode to use
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Close the file if necessary
        self.close()
        # Open the file and save the file
        self.f = open(self.fname, mode)
        # Set status
        self.closed = False
        
    # Close the file
    def close(self):
        """Close the file if it is open
        
        :Call:
            >>> q.close()
        :Inputs:
            *q*: :class:`cape.plot3d.Plot3D`
                Plot3D file interface
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Set status
        self.closed = True
        # Check if file is open
        if type(self.f).__name__ != "file":
            # Not a file!
            return
        elif self.f.closed == True:
            # File already closed
            return
        else:
            # Close the file
            self.f.close()
            
    # Read a number of integers
    def read_int(self, count=1):
        """Read *count* integers using the correct data type
        
        :Call:
            >>> I = q.read_int(count=1)
        :Inputs:
            *q*: :class:`cape.plot3d.Plot3D`
                Plot3D file interface
            *count*: :class:`int`
                Number of entries to read
        :Outputs:
            *I*: :class:`numpy.ndarray` (dtype=:class:`int` size=*count*)
                Array of *count* integers
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Check file status
        if self.closed:
            return np.array([])
        # Read data
        return np.fromfile(self.f, count=count, dtype=self.itype)
            
    # Read a number of integers
    def read_float(self, count=1):
        """Read *count* floats using the correct data type
        
        :Call:
            >>> F = q.read_float(count=1)
        :Inputs:
            *q*: :class:`cape.plot3d.Plot3D`
                Plot3D file interface
            *count*: :class:`int`
                Number of entries to read
        :Outputs:
            *F*: :class:`numpy.ndarray` (dtype=:class:`infloat` size=*count*)
                Array of *count* floats
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Check file status
        if self.closed:
            return np.array([])
        # Read data
        return np.fromfile(self.f, count=count, dtype=self.ftype)
# class Plot3D
