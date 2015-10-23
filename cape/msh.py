"""
FLUENT mesh module: :mod:`cape.msh`
===================================

This module provides the utilities for interacting with Cart3D triangulations,
including annotated triangulations (including ``.triq`` files).  Triangulations
can also be read from the UH3D format.

The module consists of individual classes that are built off of a base
triangulation class :class:`cape.tri.TriBase`.  Methods that are written for
the TriBase class apply to all other classes as well.
"""

# Required modules
# Numerics
import numpy as np
# Advanced text processing
import re
# File system and operating system management
import os, shutil

# MSH class
class Msh(object):
    """Interface for FUN3D meshes based on Fluent(R) file format
    
    :Cell types:
    
    :Faces:
        In *M.FaceCells*, the normal of *M.Face[k]* points into cell
        *M.FaceCells[k,0]* and out of cell *M.FaceCells[k,1]*
    """
    # Initialization method
    def __init__(self, fname):
        """Initialization method
        
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Dimensionality
        self.nDim = 3
        # Initialize the nodes.
        self.Nodes = np.array([])
        self.nNode = 0
        
        # Face definitions
        self.Faces = np.array([])
        self.nFace = 0
        # Connectivity
        self.FaceCells = np.array([])
        # Types
        self.Tris = np.array([])
        self.Quads = np.array([])
        
        # Cell definitions
        self.Cells = np.array([])
        self.nCell = 0
        # Cell types
        self.CellTypes = np.array([])
        # Types
        self.Prisms = np.array([])
        self.Tets   = np.array([])
        self.Pyrs   = np.array([])
        self.Hexes  = np.array([])
        
        # Face IDs
        self.FaceID = np.array([])
        # Zone types
        self.Zones = []
        
        # Read the file.
        self.ReadFluentASCII(fname)
        
    
    # Read a Fluent mesh file
    def ReadFluentASCII(self, fname):
        """Read ASCII Fluent(R) mesh file
        
        :Call:
            >>> M.ReadFluentASCII(fname)
        :Inputs:
            *M*: :class:`cape.msh.Msh`
                Volume mesh interface
            *fname*: :class:`str`
                Name of ``.msh`` file to read
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname)
        # Read the lines.
        line ='\n'
        while line != '':
            # Read the next line.
            line = f.readline()
            # Process the line type.
            typ, vals, q = self.GetFluentLineType(line)
            # Check type.
            if typ == 0 or typ == None:
                # Comment
                continue
            elif typ == 2:
                # Dimensionality
                self.nDim = vals[0]
            elif typ == 10:
                # Nodes
                if q:
                    # Closed line; overall count
                    self.nNode = vals[2]
                    self.Nodes = np.zeros((self.nNode, self.nDim))
                else:
                    # Read the nodes
                    self.ReadFluentNodesASCII(f, vals[1], vals[2])
            elif typ == 13:
                # Faces
                if q:
                    # Closed line; overall count
                    self.nFace = vals[2]
                    self.Faces = np.zeros((self.nFace, 4), dtype=int)
                    self.FaceID = np.zeros(self.nFace, dtype=int)
                    # Connectivity
                    self.FaceCells = np.zeros((self.nFace, 2), dtype=int)
                elif vals[4] == 3:
                    # Read tris
                    self.ReadFluentTrisASCII(f, vals[0], vals[1], vals[2])
                elif vals[4] == 0:
                    # Read prisms
                    self.ReadFluentMixedFacesASCII(
                        f, vals[0], vals[1], vals[2])
            elif typ == 12:
                # Cells
                if not q:
                    # Cells should not actually be spelled out, confused
                    continue
                elif vals[0] == 0:
                    # Overall count
                    self.nCell = vals[2]
                    # Initialize types
                    self.CellTypes = np.zeros(self.nCell, dtype=int)
                else:
                    # Save the types.
                    self.CellTypes[vals[1]-1:vals[2]] = vals[-1]
            elif typ == 39:
                # Zone name
                self.Zones.append(vals)
        # Close the file.
        f.close()
        
        
    # Function to process line
    def GetFluentLineType(self, line):
        """Get the line type and whether or not the line ends
        
        Entity types are tabulated below.
        
            * ``0``: comment
            * ``2``: dimensional specification
            * ``10``: nodes
            * ``12``: cells (volumes)
            * ``13``: faces
            * ``39``: zone labels
        
        :Call:
            >>> typ, vals, q = M.GetFluentLineType(line)
        :Inputs:
            *M*: :class:`cape.msh.Msh`
                Volume mesh interface
            *line*: :class:`str`
                Text from a Fluent line
        :Outputs:
            *typ*: :class:`int`
                Entity type
            *vals*: :class:`str` | :class:`list` (:class:`int`)
                List of specification indices or text if a comment
            *q*: :class:`bool`
                Whether or not line ends with ')' and is therefore closed
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Strip the line.
        line = line.strip()
        # Check the format.
        if not line.startswith('('):
            # Not a properly formatted line
            return None, None, True
        # Strip the opening parenthesis
        line = line.lstrip('(')
        # Process line ending
        if line.endswith(')'):
            # Closed line
            q = True
            line = line[:-1]
        elif line.endswith('('):
            # Open line
            q = False
            line = line[:-1]
        else:
            # Open line, no ending
            q = False
        # Split the line.
        L = line.split()
        # Get the type.
        try:
            # Read the integer.
            typ = int(L[0])
        except Exception:
            # Not a specification line, even though it starts with '('
            return None, None, True
        # Join remaining text.
        txt = ' '.join(L[1:])
        # Check for comment.
        if typ == 0:
            # Return text of string.
            vals = txt
        elif typ == 2:
            # Dimensionality; single integer
            vals = [int(txt.strip())]
        elif not txt.startswith('('):
            # Return string
            vals = txt
        elif not txt.endswith(')'):
            # Return string
            vals = txt
        elif typ == 39:
            # One index and two strings
            V = txt[1:-1].split()
            vals = [int(V[0], 16)] + V[1:]
        else:
            # List of integers
            vals = [int(v, 16) for v in txt[1:-1].split()]
        # Output
        return typ, vals, q
    
    # Function to read nodes
    def ReadFluentNodesASCII(self, f, i0, i1):
        """Read nodes from an ASCII Fluent mesh file
        
        :Call:
            >>> M.ReadFluentNodesASCII(f, i0, i1)
        :Inputs:
            *M*: :class:`cape.msh.Msh`
                Volume mesh interface
            *f*: :class:`file`
                File handle in correct location
            *i0*: :class:`int`
                Index (1-based) of first node to read
            *i1*: :class:`int`
                Index (1-based) of last node to read
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Number of nodes
        nnode = i1 - i0 + 1
        # Number of values to read
        nval = self.nDim * nnode
        # Read the data.
        A = np.fromfile(f, sep=" ", dtype=float, count=nval)
        # Check size
        if A.size != nval:
            raise IOError("Failed to read %i %iD nodes" % (nnode, self.nDim))
        # Reshape
        self.Nodes[i0-1:i1,:] = A.reshape((nnode, self.nDim))
        # Read closing parentheses
        f.readline()

    # Function to read tri faces
    def ReadFluentTrisASCII(self, f, k, i0, i1):
        """Read nodes from an ASCII Fluent mesh file
        
        :Call:
            >>> M.ReadFluentTrisASCII(f, k, i0, i1)
        :Inputs:
            *M*: :class:`cape.msh.Msh`
                Volume mesh interface
            *f*: :class:`file`
                File handle in correct location
            *k*: :class:`int`
                Component ID for these faces
            *i0*: :class:`int`
                Index (1-based) of first node to read
            *i1*: :class:`int`
                Index (1-based) of last node to read
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Number of lines
        ntri = i1 - i0 + 1
        # Read the lines
        A = np.array([
            # Split the line as a row of hex integers
            [int(v, 16) for v in f.readline().split()]
            # Loop through proper number of lines
            for i in range(ntri)
        ])
        # Check size
        if A.size != 5*ntri:
            raise IOError("Failed to read %i tris" % ntri)
        # Save the nodes
        self.Faces[i0-1:i1,:3] = A[:,:3]
        # Save the labels.
        self.FaceID[i0-1:i1] = k
        # Save the cells to which the nodes are connected
        self.FaceCells[i0-1:i1] = A[:,3:]
        # Read closing parentheses.
        f.readline()
    
    # Function to read quad faces
    def ReadFluentMixedFacesASCII(self, f, k, i0, i1):
        """Read nodes from an ASCII Fluent mesh file
        
        :Call:
            >>> M.ReadFluentQuadsASCII(f, k, i0, i1)
        :Inputs:
            *M*: :class:`cape.msh.Msh`
                Volume mesh interface
            *f*: :class:`file`
                File handle in correct location
            *k*: :class:`int`
                Component ID for these faces
            *i0*: :class:`int`
                Index (1-based) of first node to read
            *i1*: :class:`int`
                Index (1-based) of last node to read
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Number of lines
        n = i1 - i0 + 1
        # Read the lines
        A = np.array([
            # Split the line as a row of hex integers
            [int(v, 16) for v in f.readline().split()]
            # Loop through proper number of lines
            for i in range(n)
        ])
        # Check face types
        if A[0,0] == 3:
            # Triangles
            # Check size
            if A.size != 6*n:
                raise IOError("Failed to read %i tris" % n)
            # Save the nodes
            self.Faces[i0-1:i1,:3] = A[:,1:4]
        elif A[0,0] == 4:
            # Quads
            if A.size != 7*n:
                raise IOError("Failed to read %i quads" % n)
            # Save the nodes.
            self.Faces[i0-1:i1,:4] = A[:,1:5]
        # Save the labels.
        self.FaceID[i0-1:i1] = k
        # Save the cells to which the nodes are connected
        self.FaceCells[i0-1:i1] = A[:,-2:]
        # Read closing parentheses.
        f.readline()
        
    # Get the prisms
    def GetCells(self):
        """Get the volume cells from the face connectivity
        
        The results are saved to *M.Prisms* as :class:`np.ndarray`
        ((*M.nPrism*, 6), :class:`int`)
        
        :Call:
            >>> M.GetCells()
        :Inputs:
            *M*: :class:`cape.msh.Msh`
                Volume mesh interface
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Initialize the cells.
        self.Cells = np.zeros((self.nCell,8), dtype=int)
        # Loop through the faces.
        for k in np.arange(self.nFace):
            # Extract the face.
            fk = self.Faces[k]
            # Left and right cells
            jl = self.FaceCells[k,1]
            jr = self.FaceCells[k,0]
            print("k=%i, jl=%i, jr=%i, f=%s" % (k, jl, jr, fk))
            # Process
            self.ProcessFaceLR(fk, jl, 1)
            self.ProcessFaceLR(fk, jr, 0)
            
        # Split into types
        self.Prisms = self.Cells[self.CellTypes == 6]
        self.Tets   = self.Cells[self.CellTypes == 2]
        
        
    # Process one face
    def ProcessFaceLR(self, f, j, L):
        """Process the cell indices of a single face
        
        :Call:
            >>> M.ProcessFaceK(k, j)
        :Inputs:
            *M*: :class:`cape.msh.Msh`
                Volume mesh interface
            *k*: :class:`int`
                Face index
            *j*: :class:`int`
                Index for left (1) or right (0)
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Check for boundary face (only one side)
        if j == 0: return
        # Check the face type
        if f[3] == 0:
            # Tri
            n = 3
            f = f[:3]
        else:
            # Quad
            n = 4
        # Get the cell type.
        t = self.CellTypes[j-1]
        # Switch on face/cell types
        if (n==4) and (t==6):
            # Quad in prism cell; ignore
            return
        elif (n==3) and (t==6):
            # Tri in prsim cell; process
            # Check for existing information.
            if self.Cells[j-1,0] == 0:
                # Save inward normal in first slot
                if L == 0:
                    # Save inward normal as-is
                    self.Cells[j-1,0:3] = f
                else:
                    # Reverse the outward normal
                    self.Cells[j-1,0:3] = f[::-1]
            else:
                # Save outward normal in second slot.
                if L == 0:
                    # Save reverse of inward normal
                    self.Cells[j-1,3:6] = f[::-1]
                else:
                    # Save outward normal as is
                    self.Cells[j-1,3:6] = f
                    
                    
    # Line up the two tri faces of prism cells
    def RepairPrisms(self):
        """Repair prisms so that the two triangular faces are aligned
        
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        pass
        
# class Msh

