

# Required modules
import numpy as np
import os

# Triangulation class
class Tri:
    # Initialization method
    def __init__(self, nNode=None, Nodes=None, nTri=None,
        Tris=None, iComp=None):
        """
        pyCart triangulation class
        
        :Call:
            >>> tri = pyCart.Tri(nNode, Nodes, nTri, Tris, iComp)
            >>> tri = pyCart.Tri(Nodes=Nodes, Tris=Tris)
            
        :Inputs:
            *nNode*: :class:`int`
                Number of nodes in triangulation
            *Nodes*: :class:`numpy.array(dtype=float)`, (*nNode*, 3)
                Matrix of *x,y,z*-coordinates of each node
            *nTri*: :class:`int`
                Number of triangles in triangulation
            *Tris*: :class:`numpy.array(dtype=int)`, (*nTri*, 3)
                Indices of triangle vertex nodes
            *iComp*: :class:`numpy.array(dtype=int)`, (*nTri*)
                Component number for each triangle
                
        :Data members:
            Same as inputs
        
        This class provides an interface for a basic triangulation without
        surface data.
        """
        # Versions:
        #  2014.05.23 @ddalle  : First version
        
        # Check Nodes input
        if Nodes is None:
            # Initialize to an empty array.
            Nodes = np.array([])
        elif type(Nodes).__name__ != 'ndarray':
            # Attempt to convert to an array
            Nodes = np.array(Nodes)
        # Check Tris input
        if Tris is None:
            # Initialize to an empty array.
            Tris = np.array([], dtype=int)
        elif type(Tris).__name__ != 'ndarray':
            # Attempt to convert to an array
            Tris = np.array(Tris)
        # Get number of nodes from matrix
        if nNode is None:
            nNode = Nodes.shape[0]
        # Get number of nodes from matrix
        if nTri is None:
            nTri = Tris.shape[0]
        # Check for components ids
        if iComp is None:
            # Initialize to an empty array.
            iComp = np.ones(nTri)
        elif type(iComp).__name__ != 'ndarray':
            # Attempt to convert to an array
            iComp = np.array(iComp) 
        # Save the components.
        self.nNode = nNode
        self.Nodes = Nodes
        self.nTri = nTri
        self.Tris = Tris
        self.iComp = iComp
        # End
        return None
        
    # Function to write a triangulation to file.
    def Write(self, fname):
        """
        Write a triangulation to file
        
        :Call:
            >>> tri.Write(fname)
        
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
                
        :Outputs:
            ``None``
        """
        # Versions:
        #  2014.05.23 @ddalle  : First version
        
        # Open the file for creation.
        fid = open(fname, 'w')
        # Write the number of nodes and triangles.
        fid.write('%i  %i\n' % (self.nNode, self.nTri))
        # Write the nodal coordinates.
        
    
    # Function to translate the triangulation
    def Translate(self, dx=None, dy=None, dz=None):
        """
        Translate a surface triangulation
        
        :Call:
            >>> tri.Translate(dR)
            >>> tri.Translate(dx, dy, dz)
            >>> tri.Translate(dy=dy)
        
        :Inputs:
            *tri*: :class:`pyCart.tri.Tri`
                Triangulation instance to be translated
            *dR*: :class:`numpy.array` or `list`
                List of three coordinates to use for translation
            *dx*: :class:`float`
                *x*-coordinate offset
            *dy*: :class:`float`
                *y*-coordinate offset
            *dz*: :class:`float`
                *z*-coordinate offset
        
        :Outputs:
            ``None``
            
        This function translates a triangulation.  The offset coordinates may be
        specified as individual inputs or a single vector of three coordinates.
        """
        # Versions:
        #  2014.05.23 @ddalle  : First version
        
        # Check the first input type.
        if type(dx).__name__ == 'ndarray':
            # Vector
            dy = dx[1]
            dz = dx[2]
            dx = dx[0]
        else:
            # Check for unspecified inputs.
            if dx is None: dx = 0.0
            if dy is None: dy = 0.0
            if dz is None: dz = 0.0
        # Offset each coordinate.
        self.Nodes[:,0] += dx
        self.Nodes[:,1] += dy
        self.Nodes[:,2] += dz
        # End
        return None
        
        
        
        
                



# Function to read .tri files
def ReadTri(fname):
    """
    Read a basic triangulation file
    
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
    """
    # Open the file
    fid = open(fname, 'r')
    # Read the first line.
    line = fid.readline()
    # Process the line into two integers.
    nNode, nTri = (int(v) for v in line.strip().split())
    
    # Read the nodes.
    Nodes = np.fromfile(fid, dtype=float, count=nNode*3, sep=" ")
    # Reshape into a matrix.
    Nodes = Nodes.reshape((nNode,3))
    
    # Read the Tris
    Tris = np.fromfile(fid, dtype=int, count=nTri*3, sep=" ")
    # Reshape into a matrix.
    Tris = Tris.reshape((nTri,3))
    
    # Check for end of file.
    if fid.tell() == os.fstat(fid.fileno()).st_size:
        # Use default component ids.
        iComp = None
    else:
        # Read from file.
        iComp = np.fromfile(fid, dtype=int, count=nTri, sep=" ")
        
    # Create the tri object and return it.
    return Tri(nNode, Nodes, nTri, Tris, iComp)
    
    
    
