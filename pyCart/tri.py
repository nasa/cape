"""
Cart3D triangulation module: :mod:`pyCart.tri`
==============================================

This module provides the utilities for interacting with Cart3D triangulations,
including annotated triangulations (including ``.triq`` files).  Triangulations
can also be read from the UH3D format.

The module consists of individual classes that are built off of a base
triangulation class :class:`pyCart.tri.TriBase`.  Methods that are written for
the TriBase class apply to all other classes as well.
"""

# Required modules
# Numerics
import numpy as np
# File system and operating system management
import os


# Triangulation class
class TriBase(object):
    """pyCart base triangulation class
    
    This class provides an interface for a basic triangulation without
    surface data.  It can be created either by reading an ASCII file or
    specifying the data directly.
    
    When no component numbers are specified, the object created will label
    all triangles ``1``.
    
    :Call:
        >>> tri = pyCart.tri.TriBase(fname=fname)
        >>> tri = pyCart.tri.TriBase(uh3d=uh3d)
        >>> tri = pyCart.tri.TriBase(Nodes=Nodes, Tris=Tris, CompID=CompID)
    :Inputs:
        *fname*: :class:`str`
            Name of triangulation file to read (Cart3D format)
        *uh3d*: :class:`str`
            Name of triangulation file (UH3D format)
        *nNode*: :class:`int`
            Number of nodes in triangulation
        *Nodes*: :class:`numpy.array(dtype=float)`, (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *nTri*: :class:`int`
            Number of triangles in triangulation
        *Tris*: :class:`numpy.array(dtype=int)`, (*nTri*, 3)
            Indices of triangle vertex nodes
        *CompID*: :class:`numpy.array(dtype=int)`, (*nTri*)
            Component number for each triangle
    :Data members:
        *tri.nNode*: :class:`int`
            Number of nodes in triangulation
        *tri.Nodes*: :class:`numpy.array(dtype=float)`, (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *tri.nTri*: :class:`int`
            Number of triangles in triangulation
        *tri.Tris*: :class:`numpy.array(dtype=int)`, (*nTri*, 3)
            Indices of triangle vertex nodes
        *tri.CompID*: :class:`numpy.array(dtype=int)`, (*nTri*)
            Component number for each triangle
    :Versions:
        * 2014.05.23 ``@ddalle``: First version
        * 2014.06.02 ``@ddalle``: Added UH3D reading capability
    """
    # Initialization method
    def __init__(self, fname=None, uh3d=None,
        nNode=None, Nodes=None, nTri=None, Tris=None, CompID=None):
        """Initialization method"""
        # Versions:
        #  2014.05.23 @ddalle  : First version
        #  2014.06.02 @ddalle  : Added UH3D reading capability
        
        # Check if file is specified.
        if fname is not None:
            # Read from file.
            self.Read(fname)
        
        elif uh3d is not None:
            # Read from the other format.
            self.ReadUH3D(uh3d)
            
        else:
            # Process inputs.
            # Check counts.
            if nNode is None:
                # Get dimensions if possible.
                if Nodes is not None:
                    # Use the shape.
                    nNode = Nodes.shape[0]
                else:
                    # No nodes
                    nNode = 0
            # Check counts.
            if nTri is None:
                # Get dimensions if possible.
                if Tris is not None:
                    # Use the shape.
                    nTri = Tris.shape[0]
                else:
                    # No nodes
                    nTri = 0
            # Save the components.
            self.nNode = nNode
            self.Nodes = Nodes
            self.nTri = nTri
            self.Tris = Tris
            self.CompID = CompID
            
        # End
        return None
        
    # Method that shows the representation of a triangulation
    def __repr__(self):
        """Return the string representation of a triangulation.
        
        This looks like ``<pyCart.tri.Tri(nNode=M, nTri=N)>``
        
        :Versions:
            * 2014.05.27 ``@ddalle``: First version
        """
        return '<pyCart.tri.Tri(nNode=%i, nTri=%i)>' % (self.nNode, self.nTri)
        
    # String representation is the same
    __str__ = __repr__
        
    
    # Function to read node coordinates from .triq+ file
    def ReadNodes(self, f, nNode):
        """Read node coordinates from a .tri file.
        
        :Call:
            >>> tri.ReadNodes(f, nNode)
        :Inputs:
            *tri*: :class:`pyCart.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`str`
                Open file handle
            *nNode*: :class:`int`
                Number of nodes to read
        :Effects:
            Reads and creates *tri.Nodes*; file remains open.
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Save the node count.
        self.nNode = nNode
        # Read the nodes.
        Nodes = np.fromfile(f, dtype=float, count=nNode*3, sep=" ")
        # Reshape into a matrix.
        self.Nodes = Nodes.reshape((nNode,3))
        
    # Function to read triangle indices from .triq+ files
    def ReadTris(self, f, nTri):
        """Read triangle node indices from a .tri file.
        
        :Call:
            >>> tri.ReadTris(f, nTri)
        :Inputs:
            *tri*: :class:`pyCart.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`str`
                Open file handle
            *nTri*: :class:`int`
                Number of tris to read
        :Effects:
            Reads and creates *tri.Tris*; file remains open.
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Save the tri count.
        self.nTri = nTri
        # Read the Tris
        Tris = np.fromfile(f, dtype=int, count=nTri*3, sep=" ")
        # Reshape into a matrix.
        self.Tris = Tris.reshape((nTri,3))
        
    # Function to read the component identifiers
    def ReadCompID(self, f):
        """Read component IDs from a .tri file.
        
        :Call:
            >>> tri.ReadCompID(f)
        :Inputs:
            *tri*: :class:`pyCart.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`str`
                Open file handle
        :Effects:
            Reads and creates *tri.CompID* if not at end of file.  Otherwise all
            components are labeled ``1``.
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
        """
        # Check for end of file.
        if f.tell() == os.fstat(f.fileno()).st_size:
            # Use default component ids.
            self.CompID = np.ones(self.nTri)
        else:
            # Read from file.
            self.CompID = np.fromfile(f, dtype=int, count=self.nTri, sep=" ")
        
        
    # Function to read a .tri file
    def Read(self, fname):
        """Read a triangulation file (from ``*.tri``)
        
        :Call:
            >>> tri.Read(fname)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
        :Outputs:
            ``None``
        :Versions:
            * 2014.06.02 ``@ddalle``: First version
        """
        # Open the file
        fid = open(fname, 'r')
        # Read the first line.
        line = fid.readline().strip()
        # Process the line into two integers.
        nNode, nTri = (int(v) for v in line.split()[0:2])
        
        # Read the nodes.
        self.ReadNodes(fid, nNode)
        # Read the Tris.
        self.ReadTris(fid, nTri)
        # Read or assign component IDs.
        self.ReadCompID(fid)
        
        # Close the file.
        fid.close()
        
    
    # Function to write a triangulation to file.
    def Write(self, fname):
        """Write a triangulation to file
        
        :Call:
            >>> tri.Write(fname)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Examples:
            >>> tri = pyCart.ReadTri('bJet.i.tri')
            >>> tri.Write('bjet2.tri')
        :Versions:
            * 2014.05.23 ``@ddalle``: First version
        """
        # Open the file for creation.
        fid = open(fname, 'w')
        # Write the number of nodes and triangles.
        fid.write('%i  %i\n' % (self.nNode, self.nTri))
        # Write the nodal coordinates, tris, and component ids.
        np.savetxt(fid, self.Nodes, fmt="%+15.8e", delimiter=' ')
        np.savetxt(fid, self.Tris,  fmt="%i",      delimiter=' ')
        np.savetxt(fid, self.CompID, fmt="%i",      delimiter=' ')
        # Close the file.
        fid.close()
        # End
        return None
        
        
    # Function to copy a triangulation and unlink it.
    def Copy(self):
        """Copy a triangulation and unlink it
        
        :Call:
            >>> tri2 = tri.Copy()
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
        :Outputs:
            *tri2*: :class:`pyCart.tri.Tri`
                Triangulation with same values as *tri* but not linked
        :Versions:
            * 2014.06.12 ``@ddalle``: First version
        """
        # Make a new triangulation with no information.
        tri = Tri()
        # Copy over the scalars.
        tri.nNode = self.nNode
        tri.nTri  = self.nTri
        # Make new copies of the arrays.
        tri.Nodes  = self.Nodes.copy()
        tri.Tris   = self.Tris.copy()
        tri.CompID = self.CompID.copy()
        # Output the new triangulation.
        return tri
        
        
    # Read from a .uh3d file.
    def ReadUH3D(self, fname):
        """Read a triangulation file (from ``*.uh3d``)
        
        :Call:
            >>> tri.ReadUH3D(fname)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
        :Versions:
            * 2014.06.02 ``@ddalle``: First version
        """
        # Open the file
        fid = open(fname, 'r')
        # Read the first line and discard.
        line = fid.readline()
        # Read the second line and split by commas.
        data = fid.readline().split(',')
        # Process the number of nodes and tris
        nNode = int(data[0])
        nTri = int(data[2])
        # Save the statistics.
        self.nNode = nNode
        self.nTri = nTri
        
        # Initialize the nodes.
        Nodes = np.zeros((nNode, 3))
        # Loop through the nodes.
        for i in range(nNode):
            # Read the next line.
            Nodes[i] = np.fromfile(fid, dtype=float, count=4, sep=",")[1:4]
        # Save
        self.Nodes = Nodes
        
        # Initialize the Tris and component numbers
        Tris = np.zeros((nTri, 3))
        CompID = np.ones(nTri)
        # Loop through the lines.
        for i in range(nTri):
            # Read the line.
            d = np.fromfile(fid, dtype=int, count=5, sep=",")
            # Save the indices.
            Tris[i] = d[1:4]
            # Save the component number.
            CompID[i] = d[4]
        # Save.
        self.Tris = Tris
        self.CompID = CompID
        
        # Close the file.
        fid.close()
        
        
    # Get normals and areas
    def GetNormals(self):
        """Get the normals and areas of each triangle
        
        :Call:
            >>> tri.GetNormals()
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
        :Effects:
            *tri.Areas*: :class:`ndarray`, shape=(tri.nTri,)
                Area of each triangle is created
            *tri.Normals*: :class:`ndarray`, shape=(tri.nTri,3)
                Unit normal for each triangle is saved
        :Versions:
            * 201.06.12 ``@ddalle``: First version
        """
        # Extract the vertices of each tri.
        x = self.Nodes[self.Tris-1, 0]
        y = self.Nodes[self.Tris-1, 1]
        z = self.Nodes[self.Tris-1, 2]
        # Get the deltas from node 0 to node 1 or node 2
        x01 = np.vstack((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x02 = np.vstack((x[:,2]-x[:,0], y[:,2]-y[:,0], z[:,2]-z[:,0]))
        # Calculate the dimensioned normals
        n = np.cross(np.transpose(x01), np.transpose(x02))
        # Calculate the area of each triangle.
        A = np.sqrt(np.sum(n**2, 1))
        # Normalize each component.
        n[:,0] /= A
        n[:,1] /= A
        n[:,2] /= A
        # Save the areas.
        self.Areas = A/2
        # Save the unit normals.
        self.Normals = n
        # Done
        return None
       
       
    # Function to get node indices from component ID(s)
    def GetNodesFromCompID(self, i=None):
        """Find node indices based face component ID(s)
        
        :Call:
            >>> j = tri.GetNodesFromCompID(i)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *i*: :class:`int` or :class:`list` (:class:`int`)
                Component ID or list of component IDs
        :Versions:
            * 2014.09.27 ``@ddalle``: First version
        """
        # Process inputs.
        if i is None:
            # Well, this is kind of pointless.
            j = np.arange(self.nNode)
        elif np.isscalar(i):
            # Get a single component.
            J = self.Nodes(self.CompID == i)
            # Convert to unique list.
            j = np.unique(J)
        else:
            # List of components.
            J = self.Nodes(self.CompID == i[0])
            # Loop through remaining components.
            for ii in range(1,len(i)):
                # Stack the nodes from the new component.
                J = np.vstack((J, self.Nodes(self.CompID==i[0])))
            # Convert to a unique list.
            j = np.unique(J)
        # Output
        return j
        
    
    # Function to translate the triangulation
    def Translate(self, dx=None, dy=None, dz=None, i=None):
        """Translate the nodes of a triangulation object.
            
        The offset coordinates may be specified as individual inputs or a
        single vector of three coordinates.
        
        :Call:
            >>> tri.Translate(dR, i=None)
            >>> tri.Translate(dx, dy, dz, i=None)
            >>> tri.Translate(dy=dy, i=None)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *dR*: :class:`numpy.ndarray` or :class:`list`
                List of three coordinates to use for translation
            *dx*: :class:`float`
                *x*-coordinate offset
            *dy*: :class:`float`
                *y*-coordinate offset
            *dz*: :class:`float`
                *z*-coordinate offset
            *i*: :class:`int` or :class:`list` (:class:`int`0
                Component ID(s) to which to apply translation
        :Versions:
            * 2014.05.23 ``@ddalle``: First version
        """
        # Check the first input type.
        if type(dx).__name__ in ['list', 'ndarray']:
            # Vector
            dy = dx[1]
            dz = dx[2]
            dx = dx[0]
        else:
            # Check for unspecified inputs.
            if dx is None: dx = 0.0
            if dy is None: dy = 0.0
            if dz is None: dz = 0.0
        # Check for an array.
        if hasattr(dx, '__len__'):
            # Extract components
            dx, dy, dz = tuple(dx)
        # Process the node indices to be rotated.
        j = self.GetNodesFromCompID(i)
        # Offset each coordinate.
        self.Nodes[j,0] += dx
        self.Nodes[j,1] += dy
        self.Nodes[j,2] += dz
        # End
        return None
        
    # Function to rotate a triangulation about an arbitrary vector
    def Rotate(self, v1, v2, theta, i=None):
        """Rotate the nodes of a triangulation object.
        
        :Call:
            >>> tri.Rotate(v1, v2, theta)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be rotated
            *v1*: :class:`numpy.ndarray`, *shape* = (3,)
                Start point of rotation vector
            *v2*: :class:`numpy.ndarray`, *shape* = (3,)
                End point of rotation vector
            *theta*: :class:`float`
                Rotation angle in degrees
            *i*: :class:`int` or :class:`list` (:class:`int`0
                Component ID(s) to which to apply rotation
        :Versions:
            * 2014.05.27 ``@ddalle``: First version
        """
        # Convert points to NumPy.
        v1 = np.array(v1)
        v2 = np.array(v2)
        # Extract the coordinates and shift origin.
        x = self.Nodes[:,0] - v1[0]
        y = self.Nodes[:,1] - v1[1]
        z = self.Nodes[:,2] - v1[2]
        # Make the rotation vector
        v = (v2-v1) / np.linalg.linalg.norm(v2-v1)
        # Dot product of points with rotation vector
        k1 = v[0]*x + v[1]*y + v[2]*z
        # Trig functions
        c_th = np.cos(theta*np.pi/180.)
        s_th = np.sin(theta*np.pi/180.)
        # Get the indices.
        j = self.GetNodesFromCompID(i)
        # Apply Rodrigues' rotation formula to get the rotated coordinates.
        self.Nodes[j,0] = x*c_th+(v[1]*z-v[2]*y)*s_th+v[0]*k1*(1-c_th)+v1[0]
        self.Nodes[j,1] = y*c_th+(v[2]*x-v[0]*z)*s_th+v[1]*k1*(1-c_th)+v1[1]
        self.Nodes[j,2] = z*c_th+(v[0]*y-v[1]*x)*s_th+v[2]*k1*(1-c_th)+v1[2]
        # Return the rotated coordinates.
        return None
        
    # Add a second triangulation without destroying component numbers.
    def Add(self, tri):
        """Add a second triangulation file.
        
        If the new triangulation begins with a component ID less than the
        maximum component ID of the existing triangulation, the components of
        the second triangulation are offset.  For example, if both
        triangulations have components 1, 2, and 3; the IDs of the second
        triangulation, *tri2*, will be changed to 4, 5, and 6.
        
        No checks are performed, and intersections are not analyzed.
        
        :Call:
            >>> tri.Add(tri2)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be altered
            *tri2*: :class:`pyCart.tri.Tri`
                Triangulation instance to be added to the first
        :Effects:
            All nodes and triangles from *tri2* are added to *tri*.  As a
            result, the number of nodes, number of tris, and number of
            components in *tri* will all increase.
        :Versions:
            * 2014.06.12 ``@ddalle``: First version
            * 2014.10.03 ``@ddalle``: Auto detect CompID overlap
        """
        # Concatenate the node matrix.
        self.Nodes = np.vstack((self.Nodes, tri.Nodes))
        # Concatenate the triangle node index matrix.
        self.Tris = np.vstack((self.Tris, tri.Tris + self.nNode))
        # Number of components in the original triangulation
        nC = np.max(self.CompID)
        # Concatenate the component vector.
        if np.min(tri.CompID) >= nC:
            # Add the components raw (don't offset CompID.
            self.CompID = np.hstack((self.CompID, tri.CompID))
        else:
            # Adjust CompIDs to avoid overlap.
            self.CompID = np.hstack((self.CompID, tri.CompID + nC))
        # Update the statistics.
        self.nNode += tri.nNode
        self.nTri  += tri.nTri
        
    # Add a second triangulation without altering component numbers.
    def AddRawCompID(self, tri):
        """
        Add a second triangulation to the current one without changing 
        component numbers of either triangulation.  No checks are performed,
        and intersections are not analyzed.
        
        :Call:
            >>> tri.AddRawCompID(tri2)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be altered
            *tri2*: :class:`pyCart.tri.Tri`
                Triangulation instance to be added to the first
        :Effects:
            All nodes and triangles from *tri2* are added to *tri*.  As a
            result, the number of nodes, number of tris, and number of
            components in *tri* will all increase.
        :Versions:
            * 2014.06.12 ``@ddalle``: First version
        """
        # Concatenate the node matrix.
        self.Nodes = np.vstack((self.Nodes, tri.Nodes))
        # Concatenate the triangle node index matrix.
        self.Tris = np.vstack((self.Tris, tri.Tris + self.nNode))
        # Concatenate the component vector.
        self.CompID = np.hstack((self.CompID, tri.CompID))
        # Update the statistics.
        self.nNode += tri.nNode
        self.nTri  += tri.nTri
        # Done
        return None
        
    # Get normals and areas
    def GetCompArea(self, compID, n=None):
        """
        Get the total area of a component, or get the total area of a component
        projected to a plane with a given normal vector.
        
        :Call:
            >>> A = tri.GetCompArea(compID)
            >>> A = tri.GetCompArea(compID, n)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *compID*: :class:`int`
                Index of the component of which to find the area
            *n*: :class:`numpy.ndarray`
                Unit normal vector to use for projection
        :Outputs:
            *A*: :class:`float`
                Area of the component
        :Versions:
            * 2014.06.13 ``@ddalle``: First version
        """
        # Check for areas.
        if not hasattr(self, 'Areas'):
            # Calculate them.
            self.GetNormals()
        # Find the indices of tris in the component.
        i = self.CompID == compID
        # Check for direction projection.
        if n is None:
            # No projection
            return np.sum(self.Areas[i])
        else:
            # Extract the normals and copy to new matrix.
            N = self.Normals[i].copy()
            # Dot those normals with the requested vector.
            N[:,0] *= n[0]
            N[:,1] *= n[1]
            N[:,2] *= n[2]
            # Sum to get the dot product.
            d = np.sum(N, 1)
            # Multiply this dot product by the area of each tri
            return np.sum(self.Areas[i] * d)
            
    # Get normals and areas
    def GetCompNormal(self, compID):
        """Get the area-averaged unit normal of a component
        
        :Call:
            >>> n = tri.GetCompNormal(compID)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *compID*: :class:`int`
                Index of the component of which to find the normal
        :Outputs:
            *n*: :class:`numpy.ndarray` shape=(3,)
                Area-averaged unit normal
        :Versions:
            * 2014.06.13 ``@ddalle``: First version
        """
        # Check for areas.
        if not hasattr(self, 'Areas'):
            # Calculate them.
            self.GetNormals()
        # Find the indices of tris in the component.
        i = self.CompID == compID
        # Extract those normals and areas.
        N = self.Normals[i].copy()
        A = self.Areas[i].copy()
        # Weight the normals.
        N[:,0] *= A
        N[:,1] *= A
        N[:,2] *= A
        # Compute the mean.
        n = np.mean(N, 0)
        # Unitize.
        return n / np.sqrt(np.sum(n**2))
        
        
    # Function to add a bounding box based on a component and buffer
    def GetCompBBox(self, compID, **kwargs):
        """
        Find a bounding box based on the coordinates of a specified component
        or list of components, with an optional buffer or buffers in each
        direction
        
        :Call:
            >>> xlim = tri.GetCompBBox(compID, **kwargs)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *compID*: :class:`int` or :class:`list` (:class:`int`)
                Component or list of components to use for bounding box
        :Keyword arguments:
            *pad*: :class:`float`
                Buffer to add in each dimension to min and max coordinates
            *xpad*: :class:`float`
                Buffer to minimum and maximum *x*-coordinates
            *ypad*: :class:`float`
                Buffer to minimum and maximum *y*-coordinates
            *zpad*: :class:`float`
                Buffer to minimum and maximum *z*-coordinates
            *xp*: :class:`float`
                Buffer for the maximum *x*-coordinate
            *xm*: :class:`float`
                Buffer for the minimum *x*-coordinate
            *yp*: :class:`float`
                Buffer for the maximum *y*-coordinate
            *ym*: :class:`float`
                Buffer for the minimum *y*-coordinate
            *zp*: :class:`float`
                Buffer for the maximum *z*-coordinate
            *zm*: :class:`float`
                Buffer for the minimum *z*-coordinate
        :Outputs:
            *xlim*: :class:`numpy.ndarray` (:class:`float`), shape=(6,)
                List of *xmin*, *xmax*, *ymin*, *ymax*, *zmin*, *zmax*
        :Versions:
            * 2014.06.16 ``@ddalle``: First version
            * 2014.08.03 ``@ddalle``: Changed "buff" --> "pad"
        """
        # Get the overall buffer.
        pad = kwargs.get('pad', 0.0)
        # Get the other buffers.
        xpad = kwargs.get('xpad', pad)
        ypad = kwargs.get('ypad', pad)
        zpad = kwargs.get('zpad', pad)
        # Get the directional buffers.
        xp = kwargs.get('xp', xpad)
        xm = kwargs.get('xm', xpad)
        yp = kwargs.get('yp', ypad)
        ym = kwargs.get('ym', ypad)
        zp = kwargs.get('zp', zpad)
        zm = kwargs.get('zm', zpad)
        # Get the indices of the triangles to include.
        if np.isscalar(compID):
            # Single component
            i = self.CompID == compID
        else:
            # List of components; initialize with first.
            i = self.CompID == compID[0]
            # Loop through remaining components.
            for k in compID[1:]:
                i = np.logical_or(i, self.CompID == k)
        # Get the coordinates of each vertex of included tris.
        x = self.Nodes[self.Tris[i,:]-1, 0]
        y = self.Nodes[self.Tris[i,:]-1, 1]
        z = self.Nodes[self.Tris[i,:]-1, 2]
        # Get the extrema
        xmin = np.min(x) - xm
        xmax = np.max(x) + xp
        ymin = np.min(y) - ym
        ymax = np.max(y) + yp
        zmin = np.min(z) - zm
        zmax = np.max(z) + zp
        # Return the list.
        return np.array([xmin, xmax, ymin, ymax, zmin, zmax])


# Regular triangulation class
class Tri(TriBase):
    """pyCart triangulation class
    
    This class provides an interface for a basic triangulation without
    surface data.  It can be created either by reading an ASCII file or
    specifying the data directly.
    
    When no component numbers are specified, the object created will label
    all triangles ``1``.
    
    :Call:
        >>> tri = pyCart.Tri(fname=fname)
        >>> tri = pyCart.Tri(uh3d=uh3d)
        >>> tri = pyCart.Tri(Nodes=Nodes, Tris=Tris, CompID=CompID)
    :Inputs:
        *fname*: :class:`str`
            Name of triangulation file to read (Cart3D format)
        *uh3d*: :class:`str`
            Name of triangulation file (UH3D format)
        *nNode*: :class:`int`
            Number of nodes in triangulation
        *Nodes*: :class:`numpy.array(dtype=float)`, (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *nTri*: :class:`int`
            Number of triangles in triangulation
        *Tris*: :class:`numpy.array(dtype=int)`, (*nTri*, 3)
            Indices of triangle vertex nodes
        *CompID*: :class:`numpy.array(dtype=int)`, (*nTri*)
            Component number for each triangle
    :Data members:
        *tri.nNode*: :class:`int`
            Number of nodes in triangulation
        *tri.Nodes*: :class:`numpy.array(dtype=float)`, (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *tri.nTri*: :class:`int`
            Number of triangles in triangulation
        *tri.Tris*: :class:`numpy.array(dtype=int)`, (*nTri*, 3)
            Indices of triangle vertex nodes
        *tri.CompID*: :class:`numpy.array(dtype=int)`, (*nTri*)
            Component number for each triangle
    """
    
    def __init__(self, fname=None, uh3d=None,
        nNode=None, Nodes=None, nTri=None, Tris=None, CompID=None):
        """Initialization method
        
        :Versions:
            * 2014.05.23 ``@ddalle``: First version
            * 2014.06.02 ``@ddalle``: Added UH3D reading capability
        """
        # Check if file is specified.
        if fname is not None:
            # Read from file.
            self.Read(fname)
        
        elif uh3d is not None:
            # Read from the other format.
            self.ReadUH3D(uh3d)
            
        else:
            # Process inputs.
            # Check counts.
            if nNode is None:
                # Get dimensions if possible.
                if Nodes is not None:
                    # Use the shape.
                    nNode = Nodes.shape[0]
                else:
                    # No nodes
                    nNode = 0
            # Check counts.
            if nTri is None:
                # Get dimensions if possible.
                if Tris is not None:
                    # Use the shape.
                    nTri = Tris.shape[0]
                else:
                    # No nodes
                    nTri = 0
            # Save the components.
            self.nNode = nNode
            self.Nodes = Nodes
            self.nTri = nTri
            self.Tris = Tris
            self.CompID = CompID
            
        # End
        return None
        
    # Method that shows the representation of a triangulation
    def __repr__(self):
        """Return the string representation of a triangulation.
        
        This looks like ``<pyCart.tri.Tri(nNode=M, nTri=N)>``
        
        :Versions:
            8 2014.05.27 ``@ddalle``: First version
        """
        return '<pyCart.tri.Tri(nNode=%i, nTri=%i)>' % (self.nNode, self.nTri)
        
        
    # Function to read a .tri file
    def Read(self, fname):
        """Read a triangulation file (from ``*.tri``)
        
        :Call:
            >>> tri.Read(fname)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
        :Versions:
            * 2014.06.02 ``@ddalle``: split from initialization method
        """
        # Open the file
        fid = open(fname, 'r')
        # Read the first line.
        line = fid.readline().strip()
        # Process the line into two integers.
        nNode, nTri = (int(v) for v in line.split()[0:2])
        
        # Read the nodes.
        self.ReadNodes(fid, nNode)
        # Read the Tris.
        self.ReadTris(fid, nTri)
        # Read or assign component IDs.
        self.ReadCompID(fid)
        
        # Close the file.
        fid.close()


# Function to read .tri files
def ReadTri(fname):
    """Read a basic triangulation file
    
    :Call:
        >>> tri = pyCart.ReadTri(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of `.tri` file to read
    :Outputs:
        *tri*: :class:`pyCart.tri.Tri`
            Triangulation instance
    :Examples:
        >>> tri = pyCart.ReadTri('bJet.i.tri')
        >>> tri.nNode
        92852
    :Versions:
        * 2014.05.27 ``@ddalle``: First version
    """
    # Create the tri object and return it.
    return Tri(fname)
    
    
# Global function to write a triangulation (just calls tri method)
def WriteTri(fname, tri):
    """Write a triangulation instance to file
    
    :Call:
        >>> pyCart.WriteTri(fname, tri)
    :Inputs:
        *fname*: :class:`str`
            Name of `.tri` file to read
        *tri*: :class:`pyCart.tri.Tri`
            Triangulation instance
    :Examples:
        >>> tri = pyCart.ReadTri('bJet.i.tri')
        >>> pyCart.WriteTri('bjet2.tri', tri)
    :Versions:
        * 2014.05.23 ``ddalle``: First version
    """
    # Call the triangulation's write method.
    tri.Write(fname)
    return None
    
