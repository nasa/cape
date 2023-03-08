r"""
This module provides an interface to FUN3D ``.mapbc`` files, which 
specify a boundary condition and name for each component ID in the 
surface grid. An example of such a file is shown below.

    .. code-block:: none
    
               13
        21   5050        farfield_front
        22   5050        farfield_top
        23   5050        farfield_left
        24   5050        farfield_bottom
        25   5050        farfield_right
        26   5050        farfield_back
         1   4000        cap
         2   4000        body
         3   4000        base
        11   4000        fin1
        12   4000        fin2
        13   4000        fin3
        14   4000        fin4
        
The entry on the first line is the total number of components, which is
also the number of remaining rows.  Each data row has three columns:

    1. Surface component ID in original mesh
    2. FUN3D boundary condition index
    3. Name of the surface component
    
Providing an interface for this file (rather than simply copying a 
template into each run folder) is convenient because FUN3D considers 
these to be components 1 through 13 (not 21, 22, ... 14), and combining
this interface with a configuration XML file or configuration JSON file
allows users to get the index or indices of of surfaces in a FUN3D 
component by name.

If *BC* is an instance of the class provided in this module, 
:class:`MapBC`, for the ``.mapbc`` file shown above, then the following
methods show the main capabilities for going back and forth between 
component numbers and surface numbers.

    .. code-block:: pycon
    
        >>> BC.GetSurfIndex("cap")
        6
        >>> BC.GetSurfID("cap")
        7
        >>> BC.GetSurfIndex(1)
        6
        >>> BC.GetCompID("cap")
        1
        >>> BC.GerSurfID(11)
        10
        
There is also a method :func:`MapBC.SetBC` that can be used to change the
FUN3D boundary condition indices.

:See also:
    * :mod:`cape.pyfun.cntl`
    * :func:`cape.pyfun.cntl.Cntl.ReadMapBC`
    * :func:`cape.pyfun.cntl.Cntl.PrepareNamelistConfig`

"""

# Standard library
import os

# Third-party modules
import numpy as np


# MapBC class
class MapBC(object):
    r"""FUN3D boundary condition map class
    
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
        *BC.SurfID*: :class:`np.ndarray`\ [:class:`int`]
            FUN3D surface indices, numbered 1 to *n*
        *BC.CompID*: :class:`np.ndarray`\ [:class:`int`]
            Corresponding component IDs of each surface
        *BC.BCs*: :class:`np.ndarray`\ [:class:`int`]
            Boundary condition numbers
        *BC.Names*: :class:`list`\ [:class:`str`]
            List of surface names
    :Versions:
        * 2016-03-30 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname=None):
        r"""Initialization method
        
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
        r"""Representation method
        
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        return "<MapBC(n=%i)>" % self.n
    
    # Read file
    def Read(self, fname):
        r"""Read a FUN3D boundary condition map file (``.mapbc``)
        
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
            self.n = int(line.split()[0])
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
            # Safety valve
            if line == "":
                break
            # Skip empty line
            if line.strip() == "" or line.startswith("!"):
                continue
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
    def GetSurfID(self, compID, check=True, warn=False):
        r"""Get surface ID number from input component ID or name
        
        :Call:
            >>> surfID = BC.GetSurfID(compID, check=True, warn=False)
            >>> surfID = BC.GetSurfID(face, check=True, warn=False)
        :Inputs:
            *BC*: :class:`pyFun.mapbc.MapBC`
                FUN3D boundary condition interface
            *compID*: :class:`int`
                Face triangulation index
            *face*: :class:`str`
                Name of face
            *check*: {``True``} | ``False``
                Whether or not to return an error if component is not 
                found
            *warn*: ``True`` | {``False``}
                Whether or not to print warnings if not raising errors
        :Outputs:
            *surfID*: :class:`int`
                Index of the FUN3D surface, surfaces numbered 1 to *n*
        :Versions:
            * 2016-03-30 ``@ddalle``: First version
        """
        # Get the index of the entry
        surfID = self.GetSurfIndex(compID, check=check)
        # Check for a find
        if surfID is None:
            return None
        else:
            # Add one to deal with zero-based indexing
            return surfID + 1
        
    # Get surface index
    def GetSurfIndex(self, compID, check=True, warn=False):
        r"""Get surface ID number from input component ID or name
        
        :Call:
            >>> i = BC.GetSurfID(compID, check=True, warn=False)
            >>> i = BC.GetSurfID(face, check=True, warn=False)
        :Inputs:
            *BC*: :class:`pyFun.mapbc.MapBC`
                FUN3D boundary condition interface
            *compID*: :class:`int`
                Face triangulation index
            *face*: :class:`str`
                Name of face
            *check*: {``True``} | ``False``
                Whether or not to return an error if component is not 
                found
            *warn*: ``True`` | {``False``}
                Whether or not to print warnings if not raising errors
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
                # Form the error/warning message
                msg = "No surface with component ID of %s" % compID
                if check:
                    raise ValueError(msg)
                elif warn:
                    print("  Warning: " + msg)
                    return
                else:
                    return
            # Get index
            return np.where(self.CompID == compID)[0][0]
        elif t in ['str', 'unicode']:
            # Check if component name is there
            if compID not in self.Names:
                msg = "No surface found for component named '%s'" % compID
                if check:
                    raise ValueError(msg)
                elif warn:
                    print("  Warning: " + msg)
                    return
                else:
                    return
            # Get index
            return self.Names.index(compID)
        elif check or warn:
            # Unknown type
            raise TypeError("Cannot get surface ID for inputs of type '%s'"%t)
        
    # Get the component ID number
    def GetCompID(self, compID):
        r"""Get the component ID number used to tag this face in the 
        mesh
        
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
        r"""Set boundary condition
        
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
        r"""Write FUN3D MapBC file
        
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
