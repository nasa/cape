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
import os, shutil
# Specific commands to copy files and call commands.
from shutil import copy
from subprocess import call
from .util import GetTecplotCommand, TecFolder

# Attempt to load the compiled helper module.
try:
    from . import _pycart as pc
except ImportError:
    pass


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
        * 2014-05-23 ``@ddalle``: First version
        * 2014-06-02 ``@ddalle``: Added UH3D reading capability
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
            * 2014-05-27 ``@ddalle``: First version
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
            * 2014-06-16 ``@ddalle``: First version
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
            * 2014-06-16 ``@ddalle``: First version
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
            * 2014-06-16 ``@ddalle``: First version
        """
        # Check for end of file.
        if f.tell() == os.fstat(f.fileno()).st_size:
            # Use default component ids.
            self.CompID = np.ones(self.nTri)
        else:
            # Read from file.
            self.CompID = np.fromfile(f, dtype=int, count=self.nTri, sep=" ")
            
    # Function to write .tri file with one CompID per break
    def WriteVolTri(self, fname='Components.tri'):
        """Write a .tri file with one CompID per break in *tri.iTri*
        
        This is a necessary step of running `intersect` because each polyhedron
        (i.e. water-tight volume) must have a single uniform component ID before
        running `intersect`.
        
        :Call:
            >>> tri.WriteCompIDTri(fname='Components.c.tri')
        :Inputs:
            *tri*: :class:`pyCart.tri.TriBase` or derivative
                Triangulation instance
            *fname*: :class:`str`
                Name of .tri file for use as input to `intersect`
        :Versions:
            * 2015-02-24 ``@ddalle``: First version
        """
        # Copy the triangulation.
        tri = self.Copy()
        # Current maximum CompID
        comp0 = np.max(self.CompID)
        # Set first volume.
        tri.CompID[:self.iTri[0]] = comp0 + 1
        # Loop through volumes as marked in *tri.iTri*
        for k in range(len(self.iTri)-1):
            # Set the CompID for each tri in that volume.
            tri.CompID[self.iTri[k]:self.iTri[k+1]] = comp0 + k + 2
        # Write the triangulation to file.
        tri.Write(fname)
        
    # Function to map each face's CompID to the closest match from another tri
    def MapSubCompID(self, tric, compID, kc=None):
        """
        Map CompID of each face to the CompID of the nearest face in another
        triangulation.  This is a common step after running `intersect`.
        
        :Call:
            >>> tri.MapSubCompID(tric, compID, iA=0, iB=-1)
        :Inputs:
            *tri*: :class:`pyCart.tri.TriBase` or derivative
                Triangulation instance
            *tric*: :class:`pyCart.tri.TriBase` or derivative
                Triangulation with more desirable CompIDs to be copied
            *compID*: :class:`int`
                Component ID to map from *tric*
            *k1*: :class:`numpy.ndarray` (:class:`int`)
                Indices of faces in *tric* to considerider
        :Versions:
            * 2015-02-24 ``@ddalle``: First version
        """
        # Default last index.
        if kc is None: kc = np.arange(tric.nTri)
        # Indices of tris to map.
        K1 = np.where(self.CompID == compID)[0]
        # Check for a single component to map (volume really is one CompID).
        if len(np.unique(tric.CompID[kc])) == 1:
            # Map that component to each face in *k*.
            self.CompID[K1] = tric.CompID[kc[0]]
            # That's it.
            return
        # Make copy of the target indices.
        K0 = np.array(kc).copy()
        # Extract target triangle vertices
        x0 = tric.Nodes[tric.Tris[K0]-1,0]
        y0 = tric.Nodes[tric.Tris[K0]-1,1]
        z0 = tric.Nodes[tric.Tris[K0]-1,2]
        # Current vertices
        x1 = self.Nodes[self.Tris[K1]-1,0]
        y1 = self.Nodes[self.Tris[K1]-1,1]
        z1 = self.Nodes[self.Tris[K1]-1,2]
        # Length scale
        tol = 1e-6 * np.sqrt(np.sum(
            (np.max(self.Nodes,0)-np.min(self.Nodes,0))**2))
        # Start with the first tri.
        k0 = 0
        k1 = 0
        # Loop until one of the two sets of faces is exhausted.
        while (k0<len(K0)-1) and (k1<len(K1)-1):
            # Current point from intersected geometry.
            xk = x1[k1]; yk = y1[k1]; zk = z1[k1]
            # Distance to current intersected triangle.
            d0 = np.sqrt(
                (x0[k0:,0]-xk[0])**2 + (x0[k0:,1]-xk[1])**2 +
                (x0[k0:,2]-xk[2])**2 + (y0[k0:,0]-yk[0])**2 +
                (y0[k0:,1]-yk[1])**2 + (y0[k0:,2]-yk[2])**2 +
                (z0[k0:,0]-zk[0])**2 + (z0[k0:,1]-zk[1])**2 +
                (z0[k0:,2]-zk[2])**2)
            # Find the index of this tri in the target set.
            i0 = np.where(d0 <= tol)[0]
            # Check for match.
            if len(i0) == 0:
                # No match.
                k1 += 1
            else:
                # Take the first point.
                k0 += i0[0]
            # Try to match all the remaining points.
            n = min(len(K0)-k0, len(K1)-k1)
            # Calculate total of distances between vertices.
            dk = np.sqrt(np.sum((x1[k1:k1+n]-x0[k0:k0+n])**2 +
                (y1[k1:k1+n]-y0[k0:k0+n])**2 +
                (z1[k1:k1+n]-z0[k0:k0+n])**2, 1))
            # Check for a match.
            if not np.any(dk<=tol): continue
            # Find the first tri that does _not_ match.
            j = np.where(dk<=tol)[0][-1] + 1
            # Copy these *j* CompIDs.
            self.CompID[K1[k1:k1+j]] = tric.CompID[K0[k0:k0+j]]
            # Move to next tri in intersected surface.
            k1 += j; k0 += j
        
        # Find the triangles that are _still_ the old CompID
        K = np.where(self.CompID == compID)[0]
        
        # Calculate the centroids of the target components.
        x0 = np.mean(tric.Nodes[tric.Tris[K0]-1, 0], 1)
        y0 = np.mean(tric.Nodes[tric.Tris[K0]-1, 1], 1)
        z0 = np.mean(tric.Nodes[tric.Tris[K0]-1, 2], 1)
        # Calculate centroids of current tris.
        x1 = np.mean(self.Nodes[self.Tris-1,0], 1)
        y1 = np.mean(self.Nodes[self.Tris-1,1], 1)
        z1 = np.mean(self.Nodes[self.Tris-1,2], 1)
        # Loop through components.
        for i in K:
            # Find the closest centroid from *tric*.
            j = np.argmin((x0-x1[i])**2 + (y0-y1[i])**2 + (z0-z1[i])**2)
            # Map it.
            self.CompID[i] = tric.CompID[K0[j]]
            
    # Function to fully map component IDs
    def MapCompID(self, tric, tri0):
        """
        Map CompIDs from pre-intersected triangulation to an intersected
        triangulation.  In standard pyCart terminology, this is a transformation
        from :file:`Components.o.tri` to :file:`Components.i.tri`
        
        :Call:
            >>> tri.MapCompID(tric, tri0)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation interface
            *tric*: :class:`pyCart.tri.Tri`
                Full CompID breakdown prior to intersection
            *tri0*: :class:`pyCart.tri.Tri`
                Input triangulation to `intersect`
        :Versions:
            * 2015-02-25 ``@ddalle``: First version
        """
        # Get the components from the pre-intersected triangulation.
        comps = np.unique(tri0.CompID)
        # Loop through comps.
        for compID in comps:
            # Get the faces with that comp ID (before intersection)
            kc = np.where(tri0.CompID == compID)[0]
            # Map the compIDs for that component.
            self.MapSubCompID(tric, compID, kc)
        
        
    # Function to get compIDs by name
    def GetCompID(self, face=None):
        """Get components by name
        
        :Call:
            >>> compID = tri.GetCompID()
            >>> compID = tri.GetCompID(face)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation interface
            *face*: :class:`str` or :class:`int` or :class:`list`
                Component number, name, or list of component numbers and names
        :Outputs:
            *compID*: :class:`list`(:class:`int`)
                List of component IDs
        :Versions:
            * 2014-10-12 ``@ddalle``: First version
        """
        # Check input type.
        if face is None:
            # Return all components
            return list(np.unique(self.CompID))
        # Try the config.
        try:
            return self.config.GetCompID(face)
        except Exception:
            # Failed; return all components.
            return list(np.unique(self.CompID))
        
        
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
            * 2014-06-02 ``@ddalle``: First version
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
        
    # Fall-through function to write the triangulation to file.
    def Write(self, fname='Components.i.tri', v=True):
        """Write triangulation to file using fastest method available
        
        :Call:
            >>> tri.WriteSlow(fname='Components.i.tri', v=True)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
            *v*: :class:`bool`
                Whether or not
        :Examples:
            >>> tri = pyCart.ReadTri('bJet.i.tri')
            >>> tri.Write('bjet2.tri')
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
            * 2015-01-03 ``@ddalle``: Added C capability
            * 2015-02-25 ``@ddalle``: Added status update
        """
        # Status update.
        if v:
            print("     Writing triangulation: '%s'" % fname)
        # Try the fast way.
        try:
            # Fast method using compiled C.
            self.WriteFast(fname)
        except Exception:
            # Slow method using Python code.
            self.WriteSlow(fname)
    
    # Function to write a triangulation to file as fast as possible.
    def WriteFast(self, fname='Components.i.tri'):
        """Try using a compiled function to write to file
        
        :Call:
            >>> tri.WriteSlow(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2015-01-03 ``@ddalle``: First version
        """
        # See what happens.
        # Write the nodes.
        pc.WriteTri(self.Nodes, self.Tris)
        # Write the component IDs.
        pc.WriteCompID(self.CompID)
        # Check the file name.
        if fname != "Components.pyCart.tri":
            # Move the file.
            os.rename("Components.pyCart.tri", fname)
            
    
    # Function to write a triangulation to file the old-fashioned way.
    def WriteSlow(self, fname='Components.i.tri'):
        """Write a triangulation to file
        
        :Call:
            >>> tri.WriteSlow(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Examples:
            >>> tri = pyCart.ReadTri('bJet.i.tri')
            >>> tri.Write('bjet2.tri')
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
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
        
    # Function to write a UH3D file
    def WriteUH3D(self, fname='Components.i.uh3d'):
        """Write a triangulation to a UH3D file
        
        :Call:
            >>> tri.WriteUH3D(fname='Components.i.uh3d')
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Examples:
            >>> tri = pyCart.ReadTri('bJet.i.tri')
            >>> tri.WriteUH3D('bjet2.uh3d')
        :Versions:
            * 2015-04-17 ``@ddalle``: First version
        """
        # Initialize labels
        lbls = {}
        # Try to invert the configuration.
        try:
            # Loop through named components.
            for gID in self.config.faces:
                # Get the value.
                cID = self.config.GetCompID(gID)
                # Check the length.
                if len(cID) != 1: continue
                # Add it to the list.
                lbls[cID[0]] = gID
        except Exception:
            pass
        # Write the file.
        self.WriteUH3DSlow(fname, lbls)
        
        
    # Function to write a UH3D file the old-fashioned way.
    def WriteUH3DSlow(self, fname='Components.i.uh3d', lbls={}):
        """Write a triangulation to a UH3D file
        
        :Call:
            >>> tri.WriteUH3DSlow(fname='Components.i.uh3d')
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2015-04-17 ``@ddalle``: First version
        """
        # Number of component IDs
        nID = len(np.unique(self.CompID))
        # Open the file for creation.
        fid = open(fname, 'w')
        # Write the author line.
        fid.write(' file created by pyCart\n')
        # Write the information line.
        fid.write('%i, %i, %i, %i, %i, %i\n' %
            (self.nNode, self.nNode, self.nTri, self.nTri, nID, nID))
        # Loop through the nodes.
        for i in np.arange(self.nNode):
            # Write the line (with 1-based node index).
            fid.write('%i, %.12f, %.12f, %.12f\n' %
                (i+1, self.Nodes[i,0], self.Nodes[i,1], self.Nodes[i,2]))
        # Loop through the triangles.
        for k in np.arange(self.nTri):
            # Write the line (with 1-based triangle index and CompID).
            fid.write('%i, %i, %i, %i, %i\n' % (k+1, self.Tris[k,0], 
                self.Tris[k,1], self.Tris[k,0], self.CompID[k]))
        # Loop through the component names.
        for k in range(nID):
            # Get the name that will be written.
            lbl = lbls.get(k, str(k))
            # Write the label.
            fid.write("%i, '%s'\n" % (k+1, lbl))
        # Write termination line.
        fid.write('99,99,99,99,99\n')
        # Close the file.
        fid.close()
        
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
            * 2014-06-12 ``@ddalle``: First version
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
        # Try to copy the configuration list.
        try:
            tri.Conf = self.Conf.copy()
        except Exception:
            pass
        # Try to copy the configuration.
        try:
            tri.config = self.config.Copy()
        except Exception:
            pass
        # Try to copy the original barriers.
        try:
            tri.iTri = self.iTri
        except Exception:
            pass
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
            * 2014-06-02 ``@ddalle``: First version
            * 2014-10-27 ``@ddalle``: Added draft of reading component names
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
        
        # Set location.
        ftell = -1
        # Initialize components.
        Conf = {}
        # Check for named components.
        while fid.tell() != ftell:
            # Save the position.
            ftell = fid.tell()
            # Read next line.
            v = fid.readline().split(',')
            # Check if it could be a line like "'1', 'Entire'"
            if len(v) != 2: break
            # Try to convert it.
            try:
                # Get an index.
                cid = int(v[0])
                # Get the component name.
                cname = v[1].strip().strip('\'')
                # Save it.
                Conf[cname] = cid
            except Exception:
                break
        # Save the named components.
        self.Conf = Conf
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
            * 2014-06-12 ``@ddalle``: First version
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
        
    # Get edge lengths
    def GetLengths(self):
        """Get the lengths of edges
        
        :Call:
            >>> tri.GetLengths()
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
        :Effects:
            *tri.Lengths*: :class:`numpy.ndarray`, shape=(tri.nTri,3)
                Length of edge of each triangle
        :Versions:
            * 2015-02-21 ``@ddalle``: First version
        """
        # Extract the vertices of each tri.
        x = self.Nodes[self.Tris-1, 0]
        y = self.Nodes[self.Tris-1, 1]
        z = self.Nodes[self.Tris-1, 2]
        # Get the deltas from node 0->1, 1->2, 2->1
        x01 = np.vstack((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x12 = np.vstack((x[:,2]-x[:,1], y[:,2]-y[:,1], z[:,2]-z[:,1]))
        x20 = np.vstack((x[:,0]-x[:,2], y[:,0]-y[:,2], z[:,0]-z[:,2]))
        # Calculate lengths.
        self.Lengths = np.vstack((
            np.sqrt(np.sum(x01**2, 0)),
            np.sqrt(np.sum(x12**2, 0)),
            np.sqrt(np.sum(x20**2, 0)))).transpose()
        
        
    # Function to map component ID numbers to those in a Config.
    def ApplyConfig(self, cfg):
        """Change component IDs to match a configuration file
        
        Any component that is named in *tri.Conf* and *cfg.faces* has its
        component ID changed to match its intended value in *cfg*, which is an
        interface to :file:`Config.xml` files.  Note that *tri.Conf* is only
        created if the triangulation is read from a UH3D file.
        
        For example, if *tri* has a component ``'Body'`` that initially has
        component ID of 4, but the user wants that component ID to instead be
        104, then ``tri.Conf['Body']`` will be ``4``, and ``cfg.faces['Body']``
        will be ``104``.  The result of applying this method is that all faces
        in *tri.compID* that are labeled with a ``4`` will get changed to
        ``104``.
        
        This process uses a working copy of *tri* to avoid problems with the
        order of changing the component numbers.
        
        :Call:
            tri.ApplyConfig(cfg)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *cfg*: :class:`pyCart.config.Config`
                Configuration instance
        :Versions:
            * 2014-11-10 ``@ddalle``: First version
        """
        # Check for Conf in the triangulation.
        try:
            self.Conf
        except Exception:
            return
        # Make a copy of the component IDs.
        compID = self.CompID.copy()
        # Check for components.
        for k in self.Conf:
            # Check if the component is in the cfg.
            cID = cfg.faces.get(k)
            # Check for empty or list.
            if cID and (type(cID).__name__ == "int"):
                # Assign the new value.
                self.CompID[compID==self.Conf[k]] = cID
                # Save it in the Conf, too.
                self.Conf[k] = cID
       
       
    # Function to get node indices from component ID(s)
    def GetNodesFromCompID(self, i=None):
        """Find node indices from face component ID(s)
        
        :Call:
            >>> j = tri.GetNodesFromCompID(i)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *i*: :class:`int` or :class:`list` (:class:`int`)
                Component ID or list of component IDs
        :Outputs:
            *j*: :class:`numpy.array` (:class:`int`)
                Node indices, 0-based
        :Versions:
            * 2014-09-27 ``@ddalle``: First version
        """
        # Process inputs.
        if i is None:
            # Well, this is kind of pointless.
            j = np.arange(self.nNode)
        elif np.isscalar(i):
            # Get a single component.
            J = self.Tris[self.CompID == i]
            # Convert to unique list.
            j = np.unique(J) - 1
        else:
            # List of components.
            J = self.Tris[self.CompID == i[0]]
            # Loop through remaining components.
            for ii in range(1,len(i)):
                # Stack the nodes from the new component.
                J = np.vstack((J, self.Tris[self.CompID==i[ii]]))
            # Convert to a unique list.
            j = np.unique(J) - 1
        # Output
        return j
        
    # Function to get tri indices from component ID(s)
    def GetTrisFromCompID(self, i=None):
        """Find indices of triangles with specified component ID(s)
        
        :Call:
            >>> k = tri.GetTrisFromCompID(i)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *i*: :class:`int` or :class:`list` (:class:`int`)
                Component ID or list of component IDs
        :Outputs:
            *k*: :class:`numpy.ndarray` (:class:`int`)
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Process inputs.
        if i is None:
            # Return all the tris.
            return np.arange(self.nTri)
        elif i == 'entire':
            # Return all the tris.
            return np.arange(self.nTri)
        elif np.isscalar(i):
            # Get a single component.
            K = self.CompID == i
        else:
            # Initialize with all False (same size as number of tris)
            K = self.CompID < 0
            # List of components.
            for ii in i:
                # Add matches for component *ii*.
                K = np.logical_or(K, self.CompID==ii)
        # Turn boolean vector into vector of indices]
        return np.where(K)[0]
        
    # Get subtriangulation from CompID list
    def GetSubTri(self, i=None):
        """
        Get the portion of the triangulation that contains specified component
        ID(s).
        
        :Call:
            >>> tri0 = tri.GetSubTri(i=None)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *i*: :class:`int` or :class:`list` (:class:`int`)
                Component ID or list of component IDs
        :Outputs:
            *tri0*: :class:`pyCart.tri.Tri`
                Copied triangulation containing only faces with CompID in *i*
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Get the triangle indices.
        k = self.GetTrisFromCompID(i)
        # Make a copy of the triangulation.
        tri0 = self.Copy()
        # Restrict *tri0* to the matching faces.
        tri0.Tris = tri0.Tris[k]
        tri0.CompID = tri0.CompID[k]
        # Save the reduced number of tris.
        tri0.nTri = k.size
        # Output
        return tri0
        
    # Create a 3-view of a component (or list of) using TecPlot
    def Tecplot3View(self, fname, i=None):
        """Create a 3-view PNG of a component(s) using TecPlot
        
        :Call:
            >>> tri.Tecplot3View(fname, i=None)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Created file is ``'%s.png' % fname``
            *i*: :class:`str` or :class:`int` or :class:`list` (:class:`int`)
                Component name, ID or list of component IDs
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Get the subtriangulation.
        tri0 = self.GetSubTri(i)
        # Name of .tri file
        ftri = '%s.tri' % fname
        # Write triangulation file.
        tri0.Write(ftri)
        # Hide output.
        f = open('/dev/null', 'w')
        # Convert it to an STL.
        print("     Converting to STL: '%s' -> 'comp.stl'" % ftri)
        call(['tri2stl', '-i', ftri, '-o', 'comp.stl'], stdout=f)
        # Cleanup.
        for fi in ['iso-comp.mcr', 'iso-comp.lay']:
            # Check for the file.
            if os.path.isfile(fi):
                # Delete it.
                os.remove(fi)
        # Copy the template layout file and macro.
        copy(os.path.join(TecFolder, 'iso-comp.lay'), '.')
        copy(os.path.join(TecFolder, 'iso-comp.mcr'), '.')
        # Get the command for tecplot
        t360 = GetTecplotCommand()
        # Create the image.
        print("     Creating image '%s.png' using `%s`" % (fname, t360))
        call([t360, '-b', '-p', 'iso-comp.mcr'], stdout=f)
        # Close the output file.
        f.close()
        # Rename the PNG
        os.rename('iso-comp.png', '%s.png' % fname)
        # Cleanup.
        for f in ['iso-comp.mcr', 'iso-comp.lay', 'comp.stl']:
            # Check for the file.
            if os.path.isfile(f):
                # Delete it.
                os.remove(f)
                
    # Function to plot all components!
    def TecplotExplode(self):
        """
        Create a 3-view of each available named component in *tri.config* (read
        from :file:`Config.xml`) if available.  If not, create a 3-view plot for
        each *CompID*, e.g. :file:`1.png`, :file:`2.png`, etc.
        
        Call:
            >>> tri.Tecplot3View(fname, i=None)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Plot "entire.png"
        print("Plotting entire surface ...")
        print("    entire.png")
        # Create the 3-view using the name "entire" (much like Cart3D)
        self.TecPlot3View('entire', None)
        # Check for a config.
        try:
            # Appropriate status update.
            print("Plotting each named component in config ...")
            # Loop through named faces.
            for comp in self.config.faces:
                # Status update
                print("    %s.png" % comp)
                # Get the CompIDs for that face.
                k = self.config.GetCompID(comp)
                # Create the 3-view using that name.
                self.Tecplot3View(comp, k)
        except Exception:
            # Loop through CompID.
            print("FAILED.")
            print("Plotting each numbered CompID ...")
            # Loop through the available CompIDs
            for i in np.unique(self.CompID):
                # Status update.
                print("    %s.png" % i)
                # Create the 3-view plot for just that CompID==i
                self.Tecplot3View(i, i)
        
    
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
            * 2014-05-23 ``@ddalle``: First version
            * 2014-10-08 ``@ddalle``: Exported functionality to function
        """
        # Check for abort.
        if (i is None) or (i == []): return
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
        # Extract the points.
        X = self.Nodes[j,:]
        # Apply the translation.
        Y = TranslatePoints(X, [dx, dy, dz])
        # Save the translated points.
        self.Nodes[j,:] = Y
        
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
            *i*: :class:`int` or :class:`list` (:class:`int`)
                Component ID(s) to which to apply rotation
        :Versions:
            * 2014-05-27 ``@ddalle``: First version
            * 2014-10-07 ``@ddalle``: Exported functionality to function
        """
        # Check for abort.
        if (i is None) or (i == []): return
        # Get the node indices.
        j = self.GetNodesFromCompID(i)
        # Extract the points.
        X = self.Nodes[j,:]
        # Apply the rotation.
        Y = RotatePoints(X, v1, v2, theta)
        # Save the rotated points.
        self.Nodes[j,:] = Y
        
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
            * 2014-06-12 ``@ddalle``: First version
            * 2014-10-03 ``@ddalle``: Auto detect CompID overlap
        """
        # Concatenate the node matrix.
        self.Nodes = np.vstack((self.Nodes, tri.Nodes))
        # Concatenate the triangle node index matrix.
        self.Tris = np.vstack((self.Tris, tri.Tris + self.nNode))
        # Get the current component ID lists from both tries.
        CompID0 = np.unique(self.CompID)
        CompID1 = np.unique(tri.CompID)
        # Concatenate the component vector.
        if np.any(np.intersect1d(CompID0, CompID1)):
            # Number of components in the original triangulation
            nC = np.max(self.CompID)
            # Adjust CompIDs to avoid overlap.
            self.CompID = np.hstack((self.CompID, tri.CompID + nC))
        else:
            # Add the components raw (don't offset CompID.
            self.CompID = np.hstack((self.CompID, tri.CompID))
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
            * 2014-06-12 ``@ddalle``: First version
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
            * 2014-06-13 ``@ddalle``: First version
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
            * 2014-06-13 ``@ddalle``: First version
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
    def GetCompBBox(self, **kwargs):
        """
        Find a bounding box based on the coordinates of a specified component
        or list of components, with an optional buffer or buffers in each
        direction
        
        :Call:
            >>> xlim = tri.GetCompBBox(compID, **kwargs)
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance
            *compID*: :class:`int` or :class:`str` or :class:`list`
                Component or list of components to use for bounding box
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
            * 2014-06-16 ``@ddalle``: First version
            * 2014-08-03 ``@ddalle``: Changed "buff" --> "pad"
        """
        # Get the component specifier.
        face = kwargs.get('compID')
        # Process it into a list of component IDs.
        compID = self.config.GetCompID(face)
        # Quit if none specified.
        if not compID: return None
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
            * 2014-05-23 ``@ddalle``: First version
            * 2014-06-02 ``@ddalle``: Added UH3D reading capability
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
            * 2014-05-27 ``@ddalle``: First version
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
            * 2014-06-02 ``@ddalle``: split from initialization method
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
        * 2014-05-27 ``@ddalle``: First version
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
        * 2014-05-23 ``ddalle``: First version
    """
    # Call the triangulation's write method.
    tri.Write(fname)
    return None
    
# Function to rotate a triangulation about an arbitrary vector
def RotatePoints(X, v1, v2, theta):
    """Rotate a list of points
    
    :Call:
        >>> Y = RotatePoints(X, v1, v2, theta)
    :Inputs:
        *X*: :class:`numpy.ndarray`(:class:`float`), *shape* = (N,3)
            List of node coordinates
        *v1*: :class:`numpy.ndarray`, *shape* = (3,)
            Start point of rotation vector
        *v2*: :class:`numpy.ndarray`, *shape* = (3,)
            End point of rotation vector
        *theta*: :class:`float`
            Rotation angle in degrees
    :Outputs:
        *Y*: :class:`numpy.ndarray`(:class:`float`), *shape* = (N,3)
            List of rotated node coordinates
    :Versions:
        * 2014-10-07 ``@ddalle``: Copied from previous TriBase.Rotate()
    """
    # Convert points to NumPy.
    v1 = np.array(v1)
    v2 = np.array(v2)
    # Ensure array.
    if type(X).__name__ != 'ndarray':
        X = np.array(X)
    # Ensure list of points.
    if len(X.shape) == 1:
        X = np.array([X])
    # Extract the coordinates and shift origin.
    x = X[:,0] - v1[0]
    y = X[:,1] - v1[1]
    z = X[:,2] - v1[2]
    # Make the rotation vector
    v = (v2-v1) / np.linalg.linalg.norm(v2-v1)
    # Dot product of points with rotation vector
    k1 = v[0]*x + v[1]*y + v[2]*z
    # Trig functions
    c_th = np.cos(theta*np.pi/180.)
    s_th = np.sin(theta*np.pi/180.)
    # Initialize output.
    Y = X.copy()
    # Apply Rodrigues' rotation formula to get the rotated coordinates.
    Y[:,0] = x*c_th+(v[1]*z-v[2]*y)*s_th+v[0]*k1*(1-c_th)+v1[0]
    Y[:,1] = y*c_th+(v[2]*x-v[0]*z)*s_th+v[1]*k1*(1-c_th)+v1[1]
    Y[:,2] = z*c_th+(v[0]*y-v[1]*x)*s_th+v[2]*k1*(1-c_th)+v1[2]
    # Output
    return Y
    
# Function to rotate a triangulation about an arbitrary vector
def TranslatePoints(X, dR):
    """Translate the nodes of a triangulation object.
        
    The offset coordinates may be specified as individual inputs or a
    single vector of three coordinates.
    
    :Call:
        >>> TranslatePoints(X, dR)
    :Inputs:
        *X*: :class:`numpy.ndarray`(:class:`float`), *shape* = (N,3)
            List of node coordinates
        *dR*: :class:`numpy.ndarray` or :class:`list`
            List of three coordinates to use for translation
    :Outputs:
        *Y*: :class:`numpy.ndarray`(:class:`float`), *shape* = (N,3)
            List of translated node coordinates
    :Versions:
        * 2014-10-08 ``@ddalle``: Copied from previous TriBase.Translate()
    """
    # Convert points to NumPy.
    dR = np.array(dR)
    # Ensure array.
    if type(X).__name__ != 'ndarray':
        X = np.array(X)
    # Ensure list of points.
    if len(X.shape) == 1:
        X = np.array([X])
    # Initialize output.
    Y = X.copy()
    # Offset each coordinate.
    Y[:,0] += dR[0]
    Y[:,1] += dR[1]
    Y[:,2] += dR[2]
    # Output
    return Y
    
