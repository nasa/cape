"""
FLUENT mesh module: :mod:`cape.mesh`
====================================

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
class Mesh(object):
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
            *M*: :class:`cape.mesh.Mesh`
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
        
    # Write an AFLR3 mesh file
    def WriteAFLR3ASCII(self, fname):
        """Write AFLR3 mesh file
        
        :Call:
            >>> M.WriteAFLR3ASCII(fname)
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
            *fname*: :class:`str`
                Name of ``.ugrid`` file to write
        :Versions:
            * 2015-10-23 ``@ddalle``: First version
        """
        # Get the boundary zones
        kB = self.GetBoundaryZoneIDs()
        # Initialize indices of tri and quad boundary faces
        ift = (np.arange(self.nFace) < 0)
        ifq = (np.arange(self.nFace) < 0)
        # Loop through boundary faces
        for k in kB:
            # Select the faces
            fk = (self.FaceID == k)
            # Count faces on boundary.
            ift = np.logical_or(ift, np.logical_and(fk,self.Faces[:,3]==0))
            ifq = np.logical_or(ifq, np.logical_and(fk,self.Faces[:,3]>0))
        # Count faces
        ntface = np.sum(ift)
        nqface = np.sum(ifq)
        # Create array of FaceIDs for boundary faces
        ifacetag = np.hstack((self.FaceID[ift], self.FaceID[ifq]))
        # Open the file.
        f = open(fname, 'w')
        # Write number of nodes and numbers of boundary faces 
        f.write('%i %i %i ' % (self.nNode, ntface, nqface))
        # Write number of volume cells of each type
        f.write('%i %i %i %i\n' %
            (self.nTet, self.nPyr, self.nPrism, self.nHex))
        # Write the nodes.
        np.savetxt(f, self.Nodes, fmt="%20.16f", delimiter=" ")
        # Write triangular boundary faces
        if ntface > 0:
            # Downselect.
            if2nt = self.Faces[ift,:3].copy()
            # Check for left-handed tris.
            rft = (self.FaceCells[ift,0] == 0)
            if2nt[rft] = if2nt[rft,2::-1]
            # Write
            np.savetxt(f, if2nt, fmt="%i", delimiter=" ")
        # Write quad boundary 
        if nqface > 0:
            # Downselect.
            if2nq = self.Faces[ifq,:].copy()
            # Check for left-handed tris.
            rfq = (self.FaceCells[ifq,0] == 0)
            if2nq[rfq] = if2nq[rfq,3::-1]
            # Write
            np.savetxt(f, if2nq, fmt="%i", delimiter=" ")
        # Write the boundary face IDs (in a single line!)
        ifacetag.tofile(f, format="%i", sep=" ")
        f.write("\n")
        # Write the tetrahedral cells.
        if self.nTet > 0:   np.savetxt(f, self.Tets, fmt="%i", delimiter=" ")
        if self.nPyr > 0:   np.savetxt(f, self.Pyrs, fmt="%i", delimiter=" ")
        if self.nPrism > 0: np.savetxt(f, self.Prisms, fmt="%i", delimiter=" ")
        if self.nHex > 0:   np.savetxt(f, self.Hexes, fmt="%i", delimiter=" ")
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
            *M*: :class:`cape.mesh.Mesh`
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
            *M*: :class:`cape.mesh.Mesh`
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
            *M*: :class:`cape.mesh.Mesh`
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
            *M*: :class:`cape.mesh.Mesh`
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
        
        The results are saved to the following :class:`np.ndarray` arrays of
        :class:`int`\ s.
        
            * *M.Cells*:  (*M.nCell*, 8)
            * *M.Tets*:   (*M.nTet*, 4)
            * *M.Pyrs*:   (*M.nPyr*, 5)
            * *M.Prisms*: (*M.nPrism*, 6)
            * *M.Hexes*:  (*M.nHex*, 8)
        
        :Call:
            >>> M.GetCells()
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Initialize the cells.
        self.Cells = np.zeros((self.nCell,8), dtype=int)
        # Process known cell types
        self.GetPrisms()
        self.GetTets()
        self.GetPyrs()
        self.GetHexes()
        
    # Get the prisms
    def GetPrisms(self):
        """Get the prism volume cells from the face connectivity
        
        The results are saved to *M.Prisms* as :class:`np.ndarray`
        ((*M.nPrism*, 6), :class:`int`)
        
        :Call:
            >>> M.GetPrisms()
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
        :Versions:
            * 2015-10-23 ``@ddalle``: First version
        """
        # Check if the cells have been initialized
        if self.Cells.shape[0] != self.nCell:
            self.Cells = np.zeros((self.nCell,8), dtype=int)
        # Check for prisms.
        if not np.any(self.CellTypes == 6):
            self.nPrism = 0
            return
        # Loop through tri faces
        for k in np.where(self.Faces[:,3]==0)[0]:
            # Extract the face.
            fk = self.Faces[k]
            # Left and right cells
            jl = self.FaceCells[k,0]
            jr = self.FaceCells[k,1]
            # Process
            self.ProcessPrismsTri(fk, jl, 0)
            self.ProcessPrismsTri(fk, jr, 1)
        # Loop through quad faces
        for k in np.where(self.Faces[:,3]>0)[0]:
            # Extract the face.
            fk = self.Faces[k]
            # Left and right cells
            jl = self.FaceCells[k,0]
            jr = self.FaceCells[k,1]
            # Process
            self.ProcessPrismsQuad(fk, jl)
            self.ProcessPrismsQuad(fk, jr)
        # Select the prisms
        self.Prisms = self.Cells[self.CellTypes==6,:6]
        self.nPrism = self.Prisms.shape[0]
        
    # Get the tetrahedra cells
    def GetTets(self):
        """Get the tetrahedron volume cells from the face connectivity
        
        The results are saved to *M.Tets* as :class:`np.ndarray`
        ((*M.nTet*, 4), :class:`int`)
        
        :Call:
            >>> M.GetTets()
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
        :Versions:
            * 2015-10-23 ``@ddalle``: First version
        """
        # Check if the cells have been initialized
        if self.Cells.shape[0] != self.nCell:
            self.Cells = np.zeros((self.nCell,8), dtype=int)
        # Check for prisms.
        if not np.any(self.CellTypes == 2):
            self.nTet = 0
            return
        # Loop through tri faces
        for k in np.where(self.Faces[:,3]==0)[0]:
            # Extract the face.
            fk = self.Faces[k]
            # Left and right cells
            jl = self.FaceCells[k,0]
            jr = self.FaceCells[k,1]
            # Process
            self.ProcessTetsTri(fk, jl, 0)
            self.ProcessTetsTri(fk, jr, 1)
        # Select the tetrahedra
        self.Tets = self.Cells[self.CellTypes==2,:4]
        self.nTet = self.Tets.shape[0]
        
    # Get the pyramid cells
    def GetPyrs(self):
        """Get the pyramid volume cells from the face connectivity
        
        The results are saved to *M.Pyrs* as :class:`np.ndarray`
        ((*M.nPyr*, 5), :class:`int`)
        
        :Call:
            >>> M.GetPyrs()
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
        :Versions:
            * 2015-10-23 ``@ddalle``: Placeholder
        """
        # Check if the cells have been initialized
        if self.Cells.shape[0] != self.nCell:
            self.Cells = np.zeros((self.nCell,8), dtype=int)
        # Check for prisms.
        if not np.any(self.CellTypes == 5):
            self.nPyr = 0
            return
        # Process faces ...
        # Select the pyramids
        self.Pyrs = self.Cells[self.CellTypes==5,:5]
        self.nPyr = self.Pyrs.shape[0]
        
    # Get the pyramid cells
    def GetHexes(self):
        """Get the hexahedron volume cells from the face connectivity
        
        The results are saved to *M.Hexes* as :class:`np.ndarray`
        ((*M.nHex*, 8), :class:`int`)
        
        :Call:
            >>> M.GetHexe()
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
        :Versions:
            * 2015-10-23 ``@ddalle``: Placeholder
        """
        # Check if the cells have been initialized
        if self.Cells.shape[0] != self.nCell:
            self.Cells = np.zeros((self.nCell,8), dtype=int)
        # Check for prisms.
        if not np.any(self.CellTypes ==4):
            self.nHex = 0
            return
        # Process faces ...
        # Select the hexes
        self.Hexes = self.Cells[self.CellTypes==5,:8]
        self.nHex = self.Hexes.shape[0]
            
    
    # Process one face
    def ProcessPrismsTri(self, f, j, L):
        """Process the prism cell information of one tri
        
        :Call:
            >>> M.ProcessPrismsTri(f, j, L)
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
            *f*: :class:`np.ndarray` ((4), :class:`int`)
                List of vertex indices in a face (should be a tri)
            *j*: :class:`int`
                Index of neighboring cell
            *L*: :class:`int`
                Index for left (1) or right (0)
        :Versions:
            * 2015-10-22 ``@ddalle``: First version
        """
        # Check for boundary face (only one side)
        if (j == 0): return
        # Check for quad; process these in second step
        if (f[3] > 0): return
        # Get the cell type.
        t = self.CellTypes[j-1]
        # Check for prisms.
        if (t != 6): return
        # Check for existing information.
        if (self.Cells[j-1,0] > 0):
            # Already processed; fill in during quad processing
            return
        elif (L == 0):
            # Save inward normal in first slot
            self.Cells[j-1,0:3] = f[:3]
        else:
            # Save reversed outward normal in first slot.
            self.Cells[j-1,0:3] = f[2::-1]
    
    # Prepare quad contributions to prism cells
    def ProcessPrismsQuad(self, f, j):
        """Process the prism cell information of one quad
        
        :Call:
            >>> M.ProcessPrismsQuad(f, j)
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
            *f*: :class:`np.ndarray` ((4), :class:`int`)
                List of vertex indices in a face (should be a tri)
            *j*: :class:`int`
                Index of neighboring cell
        :Versions:
            * 2015-10-23 ``@ddalle``: First version
        """
        # Check for boundary face (only one side of the face in flow)
        if (j == 0): return
        # Check for tri
        if (f[3] == 0): return
        # Get the cell type
        t = self.CellTypes[j-1]
        # Check type
        if (t != 6): return
        # Process the cell
        c = self.Cells[j-1,0:6]
        # Check if it's already processed.
        if np.all(c): return
        # Find which quad nodes are in *c* layer 0 and which are in layer 1
        # Process quad node 0
        if (f[0] == c[0]):
            # Check the neighbors of f[0]
            if (f[1] == c[1]):
                # f[3],c[3]-----f[2],c[4]
                #     |             |
                # f[0],c[0]-----f[1],c[1]
                c[3] = f[3]
                c[4] = f[2]
            elif (f[1] == c[2]):
                # f[3],c[3]-----f[2],c[5]
                #     |             |
                # f[0],c[0]-----f[1],c[2]
                c[3] = f[3]
                c[5] = f[2]
            elif (f[3] == c[1]):
                # f[1],c[3]-----f[2],c[4]
                #     |             |
                # f[0],c[0]-----f[3],c[1]
                c[3] = f[1]
                c[4] = f[2]
            elif (f[3] == c[2]):
                # f[1],c[3]-----f[2],c[5]
                #     |             |
                # f[0],c[0]-----f[3],c[1]
                c[3] = f[1]
                c[5] = f[2]
        elif (f[0] == c[1]):
            # Check the neighbors of f[0]
            if (f[1] == c[2]):
                # f[3],c[4]-----f[2],c[5]
                #     |             |
                # f[0],c[1]-----f[1],c[2]
                c[4] = f[3]
                c[5] = f[2]
            elif (f[1] == c[0]):
                # f[3],c[4]-----f[2],c[3]
                #     |             |
                # f[0],c[1]-----f[1],c[0]
                c[4] = f[3]
                c[3] = f[2]
            elif (f[3] == c[2]):
                # f[1],c[4]-----f[2],c[5]
                #     |             |
                # f[0],c[1]-----f[3],c[2]
                c[4] = f[1]
                c[5] = f[2]
            elif (f[3] == c[0]):
                # f[1],c[4]-----f[2],c[3]
                #     |             |
                # f[0],c[1]-----f[3],c[0]
                c[4] = f[1]
                c[3] = f[2]
        elif (f[0] == c[2]):
            # Check the neighbors of f[0]
            if (f[1] == c[0]):
                # f[3],c[5]-----f[2],c[3]
                #     |             |
                # f[0],c[2]-----f[1],c[0]
                c[5] = f[3]
                c[3] = f[2]
            elif (f[1] == c[1]):
                # f[3],c[5]-----f[2],c[4]
                #     |             |
                # f[0],c[2]-----f[1],c[1]
                c[5] = f[3]
                c[4] = f[2]
            elif (f[3] == c[0]):
                # f[1],c[5]-----f[2],c[3]
                #     |             |
                # f[0],c[2]-----f[3],c[0]
                c[5] = f[1]
                c[3] = f[2]
            elif (f[3] == c[1]):
                # f[1],c[5]-----f[2],c[4]
                #     |             |
                # f[0],c[2]-----f[3],c[1]
                c[5] = f[1]
                c[4] = f[2]
        elif (f[2] == c[0]):
            # Check the neighbors of f[2]
            if (f[1] == c[1]):
                # f[3],c[3]-----f[0],c[4]
                #     |             |
                # f[2],c[0]-----f[1],c[1]
                c[3] = f[3]
                c[4] = f[0]
            elif (f[1] == c[2]):
                # f[3],c[3]-----f[0],c[5]
                #     |             |
                # f[2],c[0]-----f[1],c[2]
                c[3] = f[3]
                c[5] = f[0]
            elif (f[3] == c[1]):
                # f[1],c[3]-----f[0],c[4]
                #     |             |
                # f[2],c[0]-----f[3],c[1]
                c[3] = f[1]
                c[4] = f[0]
            elif (f[3] == c[2]):
                # f[1],c[3]-----f[0],c[5]
                #     |             |
                # f[2],c[0]-----f[3],c[1]
                c[3] = f[1]
                c[5] = f[0]
        elif (f[2] == c[1]):
            # Check the neighbors of f[0]
            if (f[1] == c[2]):
                # f[3],c[4]-----f[0],c[5]
                #     |             |
                # f[2],c[1]-----f[1],c[2]
                c[4] = f[3]
                c[5] = f[0]
            elif (f[1] == c[0]):
                # f[3],c[4]-----f[0],c[3]
                #     |             |
                # f[2],c[1]-----f[1],c[0]
                c[4] = f[3]
                c[3] = f[0]
            elif (f[3] == c[2]):
                # f[1],c[4]-----f[0],c[5]
                #     |             |
                # f[2],c[1]-----f[3],c[2]
                c[4] = f[1]
                c[5] = f[0]
            elif (f[3] == c[0]):
                # f[1],c[4]-----f[0],c[3]
                #     |             |
                # f[2],c[1]-----f[3],c[0]
                c[4] = f[1]
                c[3] = f[0]
        elif (f[2] == c[2]):
            # Check the neighbors of f[0]
            if (f[1] == c[0]):
                # f[3],c[5]-----f[0],c[3]
                #     |             |
                # f[2],c[2]-----f[1],c[0]
                c[5] = f[3]
                c[3] = f[0]
            elif (f[1] == c[1]):
                # f[3],c[5]-----f[0],c[4]
                #     |             |
                # f[2],c[2]-----f[1],c[1]
                c[5] = f[3]
                c[4] = f[0]
            elif (f[3] == c[0]):
                # f[1],c[5]-----f[0],c[3]
                #     |             |
                # f[2],c[2]-----f[3],c[0]
                c[5] = f[1]
                c[3] = f[0]
            elif (f[3] == c[1]):
                # f[1],c[5]-----f[0],c[4]
                #     |             |
                # f[2],c[2]-----f[3],c[1]
                c[5] = f[1]
                c[4] = f[0]
        
    # Prepare tri contributions to tet cells
    def ProcessTetsTri(self, f, j, L):
        """Process the tetrahedron cell information of one tri
        
        :Call:
            >>> M.ProcessTetsTri(f, j, L)
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
            *f*: :class:`np.ndarray` ((4), :class:`int`)
                List of vertex indices in a face (should be a tri)
            *j*: :class:`int`
                Index of neighboring cell
            *L*: :class:`int`
                Index for left (1) or right (0)
        :Versions:
            * 2015-10-23 ``@ddalle``: First version
        """
        # Check for boundary face (cell on only one side of face)
        if (j == 0): return
        # Check for quad.
        if (f[3] > 0): return
        # Get the cell type.
        t = self.CellTypes[j-1]
        # Check for tetrahedra
        if (t != 2): return
        # Extract vertices
        c = self.Cells[j-1]
        # Check for existing information
        if (c[3] > 0):
            # Already processed; continue
            return
        elif (c[0] > 0):
            # Process remaining vertices
            # Check for overlapping nodes.
            if (f[0] == c[0]):
                # Check nodes f[1] and f[2]
                if (f[1]==c[1]) or (f[1]==c[2]):
                    c[3] = f[2]
                elif (f[2]==c[1]) or (f[2]==c[2]):
                    c[3] = f[1]
            elif (f[0] == c[1]):
                # Check nodes f[1] and f[2]
                if (f[1]==c[2]) or (f[1]==c[0]):
                    c[3] = f[2]
                elif (f[2]==c[2]) or (f[2]==c[0]):
                    c[3] = f[1]
            elif (f[0] == c[2]):
                # Check nodes f[1] and f[2]
                if (f[1]==c[0]) or (f[1]==c[1]):
                    c[3] = f[2]
                elif (f[2]==c[0]) or (f[2]==c[1]):
                    c[3] = f[1]
            else:
                # Node f[0] is not in the tet yet.
                c[3] = f[0]
        elif L == 0:
            # Save nodes with inward normal
            self.Cells[j-1,:3] = f[:3]
        else:
            # Save nodes with reversed outward normal
            self.Cells[j-1,:3] = f[2::-1]
        
    # Select zones by type
    def GetZoneIDsByType(self, typs):
        """Select the zone IDs that match a list of names
        
        :Call:
            >>> K = M.GetZoneIDsByType(typs)
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
            *typs*: :class:`list` (:class:`str`)
                List of types
        :Outputs:
            *K*: :class:`list` (:class:`int`)
                List of zone IDs
        :Versions:
            * 2015-10-23 ``@ddalle``: Placeholder
        """
        # Process the zones
        return [z[0] for z in self.Zones if z[1] in typs]
    
    # Get the list of boundary zones
    def GetBoundaryZoneIDs(self):
        """Select the zone IDs that match a list of names
        
        :Call:
            >>> K = M.GetZoneIDsByType(typs)
        :Inputs:
            *M*: :class:`cape.mesh.Mesh`
                Volume mesh interface
            *typs*: :class:`list` (:class:`str`)
                List of types
        :Outputs:
            *K*: :class:`list` (:class:`int`)
                List of zone IDs
        :Versions:
            * 2015-10-23 ``@ddalle``: Placeholder
        """
        # Process the zones
        return self.GetZoneIDsByType(['wall', 'symmetry'])
        
# class Mesh

