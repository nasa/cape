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
    """FUN3D boundary condition map class
    
    :Call:
        >>> BC = MapBC(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of ``mapbc`` file to read
    :Outputs:
        *BC*: :class:`pyFun.mapbc.MapBC`
            Boundary condition map instance
        *BC.n*: :class:`int`
            Number of surfaces
        *BC.SurfID*: :class:`np.ndarray` (:class:`int`)
            FUN3D surface indices, numbered 1 to *n*
        *BC.CompID*: :class:`np.ndarray` (:class:`int`)
            Corresponding component IDs of each surface
        *BC.BCs*: :class:`np.ndarray` (:class:`int`)
            Boundary condition numbers
        *BC.Names*: :class:`list` (:class:`str`)
            List of surface names
    :Versions:
        * 2016-03-30 ``@ddalle``: First version
    """
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
            self.SurfID = np.arange(self.n)
            self.CompID = np.zeros(self.n, dtype='int')
            self.BCs = np.zeros(self.n, dtype='int')
            self.Names = []
    
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
            >>> BC.Read(fname)
        :Inputs:
            *BC*: :class:`pyFun.mapbc.MapBC`
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
    
    # Get surface ID
    def GetSurfID(self, compID):
        """Get surface ID number from input component ID or name
        
        :Call:
            >>> surfID = BC.GetSurfID(compID)
            >>> surfID = BC.GetSurfID(face)
        :Inputs:
            *BC*: :class:`pyFun.mapbc.MapBC`
                FUN3D boundary condition interface
            *compID*: :class:`int`
                Face triangulation index
            *face*: :class:`str`
                Name of face
        :Outputs:
            *surfID*: :class:`int`
                Index of the FUN3D surface, surfaces numbered 1 to *n*
        :Versions:
            * 2016-03-30 ``@ddalle``: First version
        """
        return self.GetSurfIndex(compID) + 1
        
    # Get surface index
    def GetSurfIndex(self, compID):
        """Get surface ID number from input component ID or name
        
        :Call:
            >>> i = BC.GetSurfID(compID)
            >>> i = BC.GetSurfID(face)
        :Inputs:
            *BC*: :class:`pyFun.mapbc.MapBC`
                FUN3D boundary condition interface
            *compID*: :class:`int`
                Face triangulation index
            *face*: :class:`str`
                Name of face
        :Outputs:
            *i*: :class:`int`
                Index of the FUN3D surface, 0-based
        :Versions:
            * 2016-03-30 ``@ddalle``: First version
        """
        # Check the type
        t = type(compID).__name__
        # Process which type
        if t.startswith('int'):
            # Check if the *compID* is present
            if compID not in self.CompID:
                raise ValueError(
                    "No surface with component ID of %s" % compID)
            # Get index
            return np.where(self.CompID == compID)[0][0]
        elif t in ['str', 'unicode']:
            # Check if component name is there
            if compID not in self.Names:
                raise ValueError(
                    "No surface found for name '%s'" % compID)
            # Get index
            return self.Names.index(compID)
        else:
            # Unknown type
            raise TypeError("Cannot get surface ID for inputs of type '%s'"%t)
        
    # Get the component ID number
    def GetCompID(self, compID):
        """Get the component ID number used to tag this face in the mesh
        
        :Call:
            >>> compID = BC.GetCompID(compID)
            >>> compID = BC.GetCompID(face)
        :Inputs:
            *BC*: :class:`pyFun.mapbc.MapBC`
                FUN3D boundary condition interface
            *face*: :class:`str`
                Name of face
        :Outputs:
            *compID*: :class:`int`
                Face triangulation index
        :Versions:
            * 2016-03-30 ``@ddalle``: First version
        """
        # Check the type
        t = type(compID).__name__
        # Process which type
        if t.startswith('int'):
            # Check if the *compID* is present
            if compID not in self.CompID:
                raise ValueError(
                    "No surface with component ID of %s" % compID)
            # Return index
            return compID
        elif t in ['str', 'unicode']:
            # Check if component name is there
            if compID not in self.Names:
                raise ValueError(
                    "No surface found for name '%s'" % compID)
            # Get index
            return self.CompID[self.Names.index(compID)]
        else:
            # Unknown type
            raise TypeError("Cannot get surface ID for inputs of type '%s'"%t)
            
    # Set BC
    def SetBC(self, compID, bc):
        """Set boundary condition
        
        :Call:
            >>> BC.SetBC(compID, bc)
            >>> BC.SetBC(face, bc)
        :Inputs:
            *BC*: :class:`pyFun.mapbc.MapBC`
                FUN3D boundary condition interface
            *compID*: :class:`int`
                Face triangulation index
            *face*: :class:`str`
                Name of face
            *bc*: :class:`int`
                FUN3D boundary condition number
        :Versions:
            * 2016-03-30 ``@ddalle``: First version
        """
        # Surface index
        i = self.GetSurfIndex(compID)
        # Set the boundary condition of that face
        self.BCs[i] = bc
            
    # Write the file
    def Write(self, fname=None):
        """Write FUN3D MapBC file
        
        :Call:
            >>> BC.Write(fname=None)
        :Inputs:
            *BC*: :class:`pyFun.mapbc.MapBC`
                FUN3D boundary condition interface
            *fname*: :class:`str` | ``None``
                Name of file to write; defaults to *BC.fname*
        :Versions:
            * 2016-03-30 ``@ddalle``: First version
        """
        # Get default file name
        if fname is None:
            fname = self.fname
        # Open the file
        f = open(fname, 'w')
        # Write the number of faces
        f.write('%10s%i\n' % (' ', self.n))
        # Loop through surfaces
        for i in range(self.n):
            # Write the CompID
            f.write('%7i%10i' % (self.CompID[i], self.BCs[i]))
            # Write the name
            f.write('%6s%s\n' % (' ', self.Names[i]))
        # Close file
        f.close()
    
    
# class MapBC
