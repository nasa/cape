#!/usr/bin/env python
"""
FUN3D boundary condition module: :mod:`pyFun.mapbc`
===================================================


"""

# System interface
import os
# Numerics
import numpy as np

# MapBC class
class MapBC(object):
    
    
    # Initialization method
    def __init__(self, fname=None):
        """Initialization method
        
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        # Read the file
        if fname is not None:
            # Read the file
            self.Read(fname)
        else:
            # Null BC map
            self.n = 0
    
    # Representation method(s)
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        return "<MapBC(n=%i)>" % self.n
    
    # Read file
    def Read(self, fname):
        """Read a FUN3D boundary condition map file (``.mapbc``)
        
        :Call:
            >>> map.Read(fname)
        :Inputs:
            *map*: :class:`pyFun.mapbc.MapBC`
                FUN3D boundary condition map interface
            *fname*: :class:`str`
                File name
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        # Check for the file
        if not os.path.isfile(fname):
            raise OSError("Could not find MapBC file '%s'" % fname)
        # Open the file
        f = open(fname)
        # Read the first line
        line = f.readline()
        try:
            # Get number of faces
            self.n = int(line)
        except Exception:
            raise ValueError("Could not read number of BCs from first line")
        # Initialize
        self.SurfID = np.arange(self.n)
        self.CompID = np.zeros(self.n, dtype='int')
        self.BCs = np.zeros(self.n, dtype='int')
        self.Names = []
        # Index
        i = 0
        # Loop through the remaining lines
        while i < self.n:
            # Read the line
            line = f.readline()
            # Skip empty line
            if line.strip() == "" or line.startswith("!"): continue
            # Split the components
            compID, bc, name = line.split()
            # Save values
            self.CompID[i] = int(compID)
            self.BCs[i] = int(bc)
            self.Names.append(name)
            # Increase count
            i += 1
        # Close it
        f.close()
    
# class MapBC
